import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

from nerfstudio.field_components.field_heads import FieldHeadNames


class PositionalEncoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = nn.Parameter(torch.tensor([2**i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x, y=None):
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None] ** 2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            # y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            return x_ret
        else:
            # PE
            x_ret = torch.sin(x_enc)
            return x_ret


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(True)

    def forward(self, x):
        # Extract feature pyramid from image. See Section 4.1., Section B.1 in the
        # Supplementary Materials, and: https://github.com/sxyu/pixel-nerf/blob/master/src/model/encoder.py.
        # x : B x 3 x 128 x 128

        x = self.resnet.conv1(x)  # x : B x 64 x 64 x 64
        x = self.resnet.bn1(x)  # x : B x 64 x 64 x 64
        feats1 = self.resnet.relu(x)  # feats1 : B x 64 x 64 x 64

        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))  # x : B x 64 x 32 x 32
        feats3 = self.resnet.layer2(feats2)  # x : B x 128 x 16 x 16
        feats4 = self.resnet.layer3(feats3)  # x : B x 256 x 8 x 8

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode="bilinear",
                align_corners=True,  # latents[0~3]: B x [64, 64, 128, 256] x 64 x 64
            )

        latents = torch.cat(latents, dim=1)  # B x 512 x 64 x 64
        return F.max_pool2d(latents, kernel_size=latents.size()[2:])[:, :, 0, 0]  # B x 512


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        n_blocks=8,
        n_blocks_view=1,
        skips=[4],
        z_dim=128,
        rgb_out_dim=3,
        min_deg=0,
        max_deg=16,
        viewdirs_min=0,
        viewdirs_max=4,
        ray_shape="cone",
    ):
        super().__init__()

        self.skips = skips
        self.z_dim = z_dim

        self.n_blocks = n_blocks
        self.n_blocks_view = n_blocks_view

        self.positional_encoding = PositionalEncoding(min_deg, max_deg)

        self.viewdirs_encoding = PositionalEncoding(viewdirs_min, viewdirs_max)

        self.ray_shape = ray_shape

        dim_embed = 3 * (max_deg - min_deg) * 2
        dim_embed_view = 3 * (viewdirs_max - viewdirs_min) * 2

        # Density Prediction Layers
        self.fc_in = nn.Linear(dim_embed, hidden_size)

        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.blocks = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])
        n_skips = sum([i in skips for i in range(n_blocks - 1)])

        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList([nn.Linear(z_dim, hidden_size) for i in range(n_skips)])
            self.fc_p_skips = nn.ModuleList([nn.Linear(dim_embed, hidden_size) for i in range(n_skips)])

        self.sigma_out = nn.Linear(hidden_size, 1)

        # Feature Prediction Layers
        self.fc_z_view = nn.Linear(z_dim, hidden_size)
        self.feat_view = nn.Linear(hidden_size, hidden_size)
        self.fc_view = nn.Linear(dim_embed_view, hidden_size)
        self.feat_out = nn.Linear(hidden_size, rgb_out_dim)

        self.blocks_view = nn.ModuleList(
            [nn.Linear(dim_embed_view + hidden_size, hidden_size) for _ in range(n_blocks_view - 1)]
        )

        self.fc_shape = nn.Sequential(nn.Linear(512, 128), nn.ReLU())

        self.fc_app = nn.Sequential(nn.Linear(512, 128), nn.ReLU())

    def switch_positional_encoding(self, mean, var=None, views=False):
        positional_encoding_fn = self.viewdirs_encoding if views else self.positional_encoding
        p = positional_encoding_fn(mean, var)

        return p

    def forward(self, p_in, ray_d, latent, covs=None):
        z_shape = self.fc_shape(latent)  # B x 128
        z_app = self.fc_app(latent)  # B x 128

        B, N, _ = p_in.shape  # B x (2048 * 64)

        z_shape = z_shape[:, None, :].repeat(1, N, 1)  # B x (2048 * 64) x 128
        z_app = z_app[:, None, :].repeat(1, N, 1)  # B x (2048 * 64) x 128

        p = self.switch_positional_encoding(p_in, covs)  # B x (2046 * 64) x 3 -> 96

        net = self.fc_in(p)  # B x (2048 * 64) x 128

        if z_shape is not None:
            net = net + self.fc_z(z_shape)

        net = F.relu(net)

        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            net = F.relu(layer(net))
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z_shape)
                net = net + self.fc_p_skips[skip_idx](p)
                skip_idx += 1
        sigma_out = self.sigma_out(net)

        net = self.feat_view(net)
        net = net + self.fc_z_view(z_app)

        ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
        ray_d = self.switch_positional_encoding(ray_d, views=True)
        net = net + self.fc_view(ray_d)
        net = F.relu(net)
        if self.n_blocks_view > 1:
            for layer in self.blocks_view:
                net = F.relu(layer(net))

        feat_out = self.feat_out(net)  # 12 x (2048 * 64) x 3

        return feat_out, sigma_out


class CarNeRF_Field(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ImageEncoder()
        self.decoder = Decoder()

    # def encode(self, images):
    # return self.encoder(images)

    def forward(self, xyz, latent, viewdirs=None, covs=None):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """

        rgb, sigma = self.decoder(xyz, viewdirs, latent, covs)

        outputs = {FieldHeadNames.DENSITY: F.softplus(sigma), FieldHeadNames.RGB: torch.sigmoid(rgb)}
        return outputs
