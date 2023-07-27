# import tensorflow as tf
import json

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Misc utils


def img2mse(x, y):
    return torch.mean(torch.square(x - y))


# def img2mse(x, y): return tf.reduce_mean(torch.square(x - y))


def mse2psnr(x):
    return -10.0 * torch.log(x) / torch.log(torch.tensor(10.0))


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def latentReg(z, reg):
    latentreg = 0
    for latent_i in z:
        latentreg += 1 / reg * torch.linalg.norm(latent_i)
    return latentreg


# def latentReg(z, reg): return tf.reduce_sum([1/reg * torch.linalg.norm(latent_i) for latent_i in z])


# Positional encoding
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return torch.nn.Identity(), input_dims  # TODO

    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim


# Model architecture
# def init_nerf_model(D=8, W=256, input_ch=3, input_ch_color_head=3, output_ch=4, skips=[4], use_viewdirs=False, trainable=True):

#     relu = torch.nn.ReLU()
#     def dense(W, act=relu): return torch.nn.Linear(W)    # TODO check input output
#     # def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

#     print('MODEL', input_ch, input_ch_color_head, type(
#         input_ch), type(input_ch_color_head), use_viewdirs)
#     input_ch = int(input_ch)
#     input_ch_color_head = int(input_ch_color_head)

#     # inputs = tf.keras.Input(shape=(input_ch + input_ch_color_head))
#     inputs = torch.randn(1, input_ch + input_ch_color_head)   # TODO  check
#     inputs_pts, inputs_color_head = torch.split(inputs, [input_ch, input_ch_color_head], dim=-1)
#     # inputs_pts.set_shape([None, input_ch])
#     # inputs_color_head.set_shape([None, input_ch_color_head])    #  TODO

#     print(inputs.shape, inputs_pts.shape, inputs_color_head.shape)
#     outputs = inputs_pts
#     for i in range(D):
#         outputs = dense(W)(outputs)
#         if i in skips:
#             outputs = torch.cat([inputs_pts, outputs], dim=-1)

#     if use_viewdirs:
#         alpha_out = dense(1, act=None)(outputs)
#         bottleneck = dense(256, act=None)(outputs)
#         inputs_viewdirs = torch.cat(
#             [bottleneck, inputs_color_head], dim=-1)  # concat viewdirs
#         outputs = inputs_viewdirs

#         for i in range(4):
#             outputs = dense(W//2)(outputs)
#         outputs = dense(3, act=None)(outputs)

#         outputs = torch.cat([outputs, alpha_out], dim=-1)
#     else:
#         outputs = dense(output_ch, act=None)(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)

#     if trainable == False:
#         for layer in model.layers:
#             layer.trainable = False

#     return model


class NeRF(torch.nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_color_head=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_color_head = input_ch_color_head
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)]
        )

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears = nn.ModuleList([nn.Linear(input_ch_color_head + W, W//2)] + [nn.Linear(W//2, W//2)] * 3)

        ### Implementation according to the paper
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_color_head + W, W // 2)] + [nn.Linear(W // 2, W // 2)] * (D // 2 - 1)
        )

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_color_head], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


def init_latent_vector(latent_size, name=None):
    initializer = torch.randn(latent_size, dtype=torch.float32)
    initializer.requires_grad = True
    # initializer = tf.random_normal_initializer(mean=0., stddev=0.01)
    return initializer

    # return tf.Variable(initializer(shape=[latent_size], dtype=tf.float32),
    #                    trainable=True,
    #                    validate_shape=True,
    #                    name=name)


# Ray helpers
def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    # Tensorflow version
    i, j = torch.meshgrid(
        torch.arange(0, W, dtype=torch.float32), torch.arange(0, H, dtype=torch.float32), indexing="xy"
    )
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], dim=-1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = torch.broadcast_to(torch.Tensor(c2w[:3, -1]), rays_d.shape)
    # rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    # Numpy Version
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (W / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1.0 / (H / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling helper


def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights += 1e-5  # prevent nans

    pdf = weights / torch.sum(weights, dim=-1, keepdims=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    # cdf = tf.cumsum(pdf, -1)
    # cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, N_samples)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand_like(list(cdf.shape[:-1]) + [N_samples])  # TODO
        # u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = torch.searchsorted(cdf, u, side="right")
    below = torch.maximum(0, inds - 1)
    above = torch.minimum(cdf.shape[-1] - 1, inds)
    inds_g = torch.stack([below, above], dim=-1)
    cdf_g = torch.gather(cdf, -1, inds_g)
    bins_g = torch.gather(bins, -1, inds_g)
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


# Plane-Ray intersection helper
def plane_pts(rays, planes, id_planes, near, method="planes"):
    """Ray-Plane intersection for given planes in the scene

    Args:
        rays: ray origin and directions
        planes: first plane position, plane normal and distance between planes
        id_planes: ids of used planes
        near: distance between camera pose and first intersecting plane
        method: Method used

    Returns:
        pts: [N_rays, N_samples+N_importance] - intersection points of rays and selected planes
        z_vals: position of the point along each ray respectively
    """
    # Extract ray and plane definitions
    rays_o, rays_d = rays
    N_rays = rays_o.shape[0]
    # N_rays = rays_o.get_shape().as_list()[0]
    plane_bds, plane_normal, delta = planes
    plane_bds = torch.from_numpy(plane_bds).to(torch.float32)
    plane_normal = torch.from_numpy(plane_normal).to(torch.float32)

    # Get amount of all planes
    n_planes = torch.ceil(torch.linalg.norm(plane_bds[:, -1] - plane_bds[:, 0]) / delta) + 1
    # n_planes = tf.math.ceil(tf.norm(plane_bds[:, -1] - plane_bds[:, 0]) / delta) + 1

    # Calculate how far the ray_origins lies apart from each plane
    d_ray_first_plane = torch.matmul(plane_bds[:, 0] - rays_o, plane_normal[:, None])
    d_ray_first_plane = torch.maximum(-d_ray_first_plane, -near)
    # d_ray_first_plane = tf.matmul(plane_bds[:, 0]-rays_o, plane_normal[:, None])
    # d_ray_first_plane = tf.maximum(-d_ray_first_plane, -near)

    # Get the ids of the planes in front of each ray starting from near distance upto the far plane
    start_id = torch.ceil((d_ray_first_plane + near) / delta)
    plane_id = start_id + torch.from_numpy(id_planes).to(torch.float32)
    if method == "planes":
        plane_id = torch.cat([plane_id[:, :-1], n_planes.repeat_interleave(N_rays)[:, None]], dim=1)
        # plane_id = tf.concat([plane_id[:, :-1], tf.repeat(n_planes, N_rays)[:, None]], axis=1)
    elif method == "planes_plus":
        # Experimental setup, that got discarded due to lower or the same quality
        plane_id = torch.cat(
            [
                plane_id[:, :1],
                id_planes[None, 1:-1].repeat_interleave(N_rays, dim=0),
                n_planes.repeat_interleave(N_rays)[:, None],
            ],
            dim=1,
        )
        # plane_id = tf.concat([plane_id[:, :1],
        #                       tf.repeat(id_planes[None, 1:-1], N_rays, axis=0),
        #                       tf.repeat(n_planes, N_rays)[:, None]], axis=1)

    # [N_samples, N_rays, xyz]
    z_planes = plane_normal[None, None, :] * torch.transpose(plane_id * delta, 0, 1)[..., None]
    # z_planes = plane_normal[None, None, :] * tf.transpose(plane_id*delta)[..., None]
    relevant_plane_origins = plane_bds[:, 0][None, None, :] + z_planes

    # Distance between each ray's origin and associated planes
    d_plane_pose = relevant_plane_origins - rays_o[None, :, :]

    n = torch.matmul(d_plane_pose, plane_normal[..., None])
    z = torch.matmul(rays_d, plane_normal[..., None])

    z_vals = torch.transpose(torch.squeeze(n / z), 0, 1)

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]

    return pts, z_vals


def rotate_yaw(p, yaw):
    """Rotates p with yaw in the given coord frame with y being the relevant axis and pointing downwards

    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        yaw: Rotation angle

    Returns:
        p: Rotated points [N_pts, N_frames, N_samples, 3]
    """
    # p of size [batch_rays, n_obj, samples, xyz]
    if len(p.shape) < 4:
        # p = p[..., tf.newaxis, :]
        p = p.unsqueeze(-2)

    c_y = torch.cos(yaw)
    s_y = torch.sin(yaw)

    if len(c_y.shape) < 3:
        c_y = c_y.unsqueeze(-1)
        s_y = s_y.unsqueeze(-1)

    # c_y = tf.cos(yaw)[..., tf.newaxis]
    # s_y = tf.sin(yaw)[..., tf.newaxis]

    p_x = c_y * p[..., 0] - s_y * p[..., 2]
    p_y = p[..., 1]
    p_z = s_y * p[..., 0] + c_y * p[..., 2]

    # return tf.concat([p_x[..., tf.newaxis], p_y[..., tf.newaxis], p_z[..., tf.newaxis]], axis=-1)
    return torch.cat([p_x.unsqueeze(-1), p_y.unsqueeze(-1), p_z.unsqueeze(-1)], dim=-1)


def scale_frames(p, sc_factor, inverse=False):
    """Scales points given in N_frames in each dimension [xyz] for each frame or rescales for inverse==True

    Args:
        p: Points given in N_frames frames [N_points, N_frames, N_samples, 3]
        sc_factor: Scaling factor for new frame [N_points, N_frames, 3]
        inverse: Inverse scaling if true, bool

    Returns:
        p_scaled: Points given in N_frames rescaled frames [N_points, N_frames, N_samples, 3]
    """
    # Take 150% of bbox to include shadows etc.
    dim = torch.tensor([1.0, 1.0, 1.0]).to(sc_factor.device) * sc_factor
    # dim = tf.constant([0.1, 0.1, 0.1]) * sc_factor

    half_dim = dim / 2
    scaling_factor = (1 / (half_dim + 1e-9)).unsqueeze(-2)
    # scaling_factor = (1 / (half_dim + 1e-9))[:, :, tf.newaxis, :]

    if not inverse:
        p_scaled = scaling_factor * p
    else:
        p_scaled = (1.0 / scaling_factor) * p

    return p_scaled


def world2object(pts, dirs, pose, theta_y, dim=None, inverse=False):
    """Transform points given in world frame into N_obj object frames

    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim

    Args:
        pts: N_pts times 3D points given in world frame, [N_pts, 3]
        dirs: Corresponding 3D directions given in world frame, [N_pts, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        inverse: if true pts and dirs should be given in object frame and are transofmed back into world frame, bool
            For inverse: pts, [N_pts, N_obj, 3]; dirs, [N_pts, N_obj, 3]

    Returns:
        pts_w: 3d points transformed into object frame (world frame for inverse task)
        dir_w: unit - 3d directions transformed into object frame (world frame for inverse task)
    """

    #  Prepare args if just one sample per ray-object or world frame only
    if len(pts.shape) == 3:
        # [batch_rays, n_obj, samples, xyz]
        n_sample_per_ray = pts.shape[1]

        pose = pose.repeat_interleave(n_sample_per_ray, dim=0)
        theta_y = theta_y.repeat_interleave(n_sample_per_ray, dim=0)
        if dim is not None:
            dim = dim.repeat_interleave(n_sample_per_ray, dim=0)
        if len(dirs.shape) == 2:
            dirs = dirs.repeat_interleave(n_sample_per_ray, dim=0)

        # pose = tf.repeat(pose, n_sample_per_ray, axis=0)
        # theta_y = tf.repeat(theta_y, n_sample_per_ray, axis=0)
        # if dim is not None:
        #     dim = tf.repeat(dim, n_sample_per_ray, axis=0)
        # if len(dirs.shape) == 2:
        #     dirs = tf.repeat(dirs, n_sample_per_ray, axis=0)

        pts = torch.tensor.reshape(pts, [-1, 3])

    # Shift the object reference point to the middle of the bbox (vkitti2 specific)
    y_shift = (
        torch.tensor([0.0, -1.0, 0.0]).unsqueeze(0)
        if inverse
        else torch.tensor([0.0, -1.0, 0.0]).unsqueeze(0).unsqueeze(0)
    ).to(dim.device) * (dim[..., 1] / 2).unsqueeze(-1)
    # y_shift = (tf.constant([0., -1., 0.])[tf.newaxis, :] if inverse else
    #            tf.constant([0., -1., 0.])[tf.newaxis, tf.newaxis, :]) * \
    #           (dim[..., 1] / 2)[..., tf.newaxis]
    pose_w = pose + y_shift

    # Describes the origin of the world system w in the object system o
    t_w_o = rotate_yaw(-pose_w, theta_y)

    if not inverse:
        N_obj = theta_y.shape[1]
        pts_w = pts.unsqueeze(1).repeat_interleave(N_obj, dim=1)
        dirs_w = dirs.unsqueeze(1).repeat_interleave(N_obj, dim=1)
        # pts_w = tf.repeat(pts[:, tf.newaxis, ...], N_obj, axis=1)
        # dirs_w = tf.repeat(dirs[:, tf.newaxis, ...], N_obj, axis=1)

        # Rotate coordinate axis
        # TODO: Generalize for 3d roaations
        pts_o = rotate_yaw(pts_w, theta_y) + t_w_o
        dirs_o = rotate_yaw(dirs_w, theta_y)

        # Scale rays_o_v and rays_d_v for box [[-1.,1], [-1.,1], [-1.,1]]
        if dim is not None:
            pts_o = scale_frames(pts_o, dim)
            dirs_o = scale_frames(dirs_o, dim)

        # Normalize direction
        dirs_o = dirs_o / torch.linalg.norm(dirs_o, dim=3).unsqueeze(-2)
        # dirs_o = dirs_o / tf.norm(dirs_o, axis=3)[..., tf.newaxis, :]
        return [pts_o, dirs_o]

    else:
        pts_o = pts.unsqueeze(1).unsqueeze(0)
        # pts_o = pts[tf.newaxis, :, tf.newaxis, :]
        dirs_o = dirs
        if dim is not None:
            pts_o = scale_frames(pts_o, dim.unsqueeze(0), inverse=True)
            # pts_o = scale_frames(pts_o, dim[tf.newaxis, ...], inverse=True)
            if dirs is not None:
                dirs_o = scale_frames(dirs_o, dim, inverse=True)

        pts_o = pts_o - t_w_o
        pts_w = rotate_yaw(pts_o, -theta_y)[0, :]

        if dirs is not None:
            dirs_w = rotate_yaw(dirs_o, -theta_y)
            # Normalize direction
            dirs_w = dirs_w / torch.linalg.norm(dirs_w, dim=-1).unsqueeze(-2)
            # dirs_w = dirs_w / tf.norm(dirs_w, axis=-1)[..., tf.newaxis, :]
        else:
            dirs_w = None

        return [pts_w, dirs_w]


def object2world(pts, dirs, pose, theta_y, dim=None, inverse=True):
    """Transform points given in world frame into N_obj object frames

    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim

    Args:
        pts: N_pts times 3D points given in N_obj object frames, [N_pts, N_obj, 3]
        dirs: Corresponding 3D directions given in N_obj object frames, [N_pts, N_obj, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]

    Returns:
        pts_w: 3d points transformed into world frame
        dir_w: unit - 3d directions transformed into world frame
    """

    #  Prepare args if just one sample per ray-object
    if len(pts.shape) == 3:
        # [N_rays, N_obj, N_obj_samples, xyz]
        n_sample_per_ray = pts.shape[1]

        pose = pose.repeat_interleave(n_sample_per_ray, dim=0)
        theta_y = theta_y.repeat_interleave(n_sample_per_ray, dim=0)
        # pose = tf.repeat(pose, n_sample_per_ray, axis=0)
        # theta_y = tf.repeat(theta_y, n_sample_per_ray, axis=0)

        if dim is not None:
            dim = dim.repeat_interleave(n_sample_per_ray, dim=0)
            # dim = tf.repeat(dim, n_sample_per_ray, axis=0)
        if len(dirs.shape) == 2:
            dirs = dirs.repeat_interleave(n_sample_per_ray, dim=0)
            # dirs = tf.repeat(dirs, n_sample_per_ray, axis=0)

        pts = torch.reshape(pts, [-1, 3])

    # Shift the object reference point to the middle of the bbox (vkitti2 specific)
    y_shift = torch.tensor([0.0, -1.0, 0.0]).unsqueeze(0).to(dim.device) * (dim[..., 1] / 2).unsqueeze(-1)
    # y_shift = tf.constant([0., -1., 0.])[tf.newaxis, :] * (dim[..., 1] / 2)[..., tf.newaxis]
    pose_w = pose + y_shift

    # Describes the origin of the world system w in the object system o
    t_w_o = rotate_yaw(-pose_w, theta_y)

    pts_o = pts.unsqueeze(-2).unsqueeze(0)
    dirs_o = dirs
    if dim is not None:
        pts_o = scale_frames(pts_o, dim.unsqueeze(0), inverse=True)
        # pts_o = scale_frames(pts_o, dim[tf.newaxis, ...], inverse=True)
        if dirs is not None:
            dirs_o = scale_frames(dirs_o, dim, inverse=True)

    pts_o = pts_o - t_w_o
    pts_w = rotate_yaw(pts_o, -theta_y)[0, :]

    if dirs is not None:
        dirs_w = rotate_yaw(dirs_o, -theta_y)
        # Normalize direction
        dirs_w = dirs_w / torch.linalg.norm(dirs_w, dim=-1).unsqueeze(-2)
        # dirs_w = dirs_w / tf.norm(dirs_w, axis=-1)[..., tf.newaxis, :]
    else:
        dirs_w = None

    return [pts_w, dirs_w]


def ray_box_intersection(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected

    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary

    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified

    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = torch.ones_like(ray_o) * -1.0  # tf.constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = torch.ones_like(ray_o)  # tf.constant([1., 1., 1.])
    # if aabb_min is None:
    #     aabb_min = tf.ones_like(ray_o) * -1. # tf.constant([-1., -1., -1.])
    # if aabb_max is None:
    #     aabb_max = tf.ones_like(ray_o) # tf.constant([1., 1., 1.])

    inv_d = torch.reciprocal(ray_d)

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = torch.minimum(t_min, t_max)
    t1 = torch.maximum(t_min, t_max)

    t_near = torch.maximum(torch.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = torch.minimum(torch.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])

    # Check if rays are inside boxes
    intersection_map = t_far > t_near  # TODO check dim
    # Check that boxes are in front of the ray origin
    positive_far = t_far > 0

    intersection_map = torch.stack(list(torch.where(torch.logical_and(intersection_map, positive_far))), dim=1)

    # intersection_map = tf.where(t_far > t_near)
    # # Check that boxes are in front of the ray origin
    # positive_far = tf.where(tf.gather_nd(t_far, intersection_map) > 0)
    # intersection_map = tf.gather_nd(intersection_map, positive_far)

    # if not intersection_map.shape[0] == 0:
    #     z_ray_in = tf.gather_nd(t_near, intersection_map)
    #     z_ray_out = tf.gather_nd(t_far, intersection_map)
    if not intersection_map.shape[0] == 0:
        z_ray_in = t_near[intersection_map[:, 0], intersection_map[:, 1]]
        z_ray_out = t_far[intersection_map[:, 0], intersection_map[:, 1]]
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def box_pts(rays, pose, theta_y, dim=None, one_intersec_per_ray=False):
    """gets ray-box intersection points in world and object frames in a sparse notation

    Args:
        rays: ray origins and directions, [[N_rays, 3], [N_rays, 3]]
        pose: object positions in world frame for each ray, [N_rays, N_obj, 3]
        theta_y: rotation of objects around world y axis, [N_rays, N_obj]
        dim: object bounding box dimensions [N_rays, N_obj, 3]
        one_intersec_per_ray: If True only the first interesection along a ray will lead to an
        intersection point output

    Returns:
        pts_box_w: box-ray intersection points given in the world frame
        viewdirs_box_w: view directions of each intersection point in the world frame
        pts_box_o: box-ray intersection points given in the respective object frame
        viewdirs_box_o: view directions of each intersection point in the respective object frame
        z_vals_w: integration step in the world frame
        z_vals_o: integration step for scaled rays in the object frame
        intersection_map: mapping of points, viewdirs and z_vals to the specific rays and objects at the intersection
        rays_o_o: rays_o in object frame

    """
    rays_o, rays_d = rays
    pose = pose.to(rays_o.device)
    theta_y = theta_y.to(rays_o.device)
    if dim is not None:
        dim = dim.to(rays_o.device)
    # Transform each ray into each object frame
    rays_o_o, dirs_o = world2object(rays_o, rays_d, pose, theta_y, dim)
    rays_o_o = torch.squeeze(rays_o_o)
    dirs_o = torch.squeeze(dirs_o)

    # Get the intersection with each Bounding Box
    z_ray_in_o, z_ray_out_o, intersection_map = ray_box_intersection(rays_o_o, dirs_o)

    if z_ray_in_o is not None:
        # Calculate the intersection points for each box in each object frame
        pts_box_in_o = (
            rays_o_o[intersection_map[:, 0], intersection_map[:, 1], :]
            + z_ray_in_o.unsqueeze(-1) * dirs_o[intersection_map[:, 0], intersection_map[:, 1], :]
        )
        # Transform the intersection points for each box in world frame
        pts_box_in_w, _ = object2world(
            pts_box_in_o,
            None,
            pose[intersection_map[:, 0], intersection_map[:, 1], :],
            theta_y[intersection_map[:, 0], intersection_map[:, 1]],
            dim[intersection_map[:, 0], intersection_map[:, 1], :],
        )
        pts_box_in_w = torch.squeeze(pts_box_in_w)

        # Get all intersecting rays in unit length and the corresponding z_vals
        rays_o_in_w = rays_o.unsqueeze(-2).repeat_interleave(pose.shape[1], dim=1)[
            intersection_map[:, 0], intersection_map[:, 1], :
        ]
        rays_d_in_w = rays_d.unsqueeze(-2).repeat_interleave(pose.shape[1], dim=1)[
            intersection_map[:, 0], intersection_map[:, 1], :
        ]
        # rays_o_in_w = tf.gather_nd(tf.repeat(rays_o[:, tf.newaxis, :], pose.shape[1], axis=1), intersection_map)
        # rays_d_in_w = tf.gather_nd(tf.repeat(rays_d[:, tf.newaxis, :], pose.shape[1], axis=1), intersection_map)
        # Account for non-unit length rays direction
        z_vals_in_w = torch.linalg.norm(pts_box_in_w - rays_o_in_w, dim=1) / torch.linalg.norm(rays_d_in_w, dim=-1)

        if one_intersec_per_ray:
            # Get just nearest object point on a single ray
            z_vals_in_w, intersection_map, first_in_only = get_closest_intersections(
                z_vals_in_w, intersection_map, N_rays=rays_o.shape[0], N_obj=theta_y.shape[1]
            )
            # Get previous calculated values just for first intersections
            z_ray_in_o = z_ray_in_o[first_in_only]
            z_ray_out_o = z_ray_out_o[first_in_only]
            pts_box_in_o = pts_box_in_o[first_in_only]
            pts_box_in_w = pts_box_in_w[first_in_only]
            rays_o_in_w = rays_o_in_w[first_in_only]
            rays_d_in_w = rays_d_in_w[first_in_only]

        # Get the far intersection points and integration steps for each ray-box intersection in world and object frames
        pts_box_out_o = (
            rays_o_o[intersection_map[:, 0], intersection_map[:, 1], :]
            + z_ray_out_o.unsqueeze(-1) * dirs_o[intersection_map[:, 0], intersection_map[:, 1], :]
        )

        pts_box_out_w, _ = object2world(
            pts_box_out_o,
            None,
            pose[intersection_map[:, 0], intersection_map[:, 1], :],
            theta_y[intersection_map[:, 0], intersection_map[:, 1]],
            dim[intersection_map[:, 0], intersection_map[:, 1], :],
        )
        pts_box_out_w = torch.squeeze(pts_box_out_w)
        z_vals_out_w = torch.linalg.norm(pts_box_out_w - rays_o_in_w, dim=1) / torch.linalg.norm(rays_d_in_w, dim=-1)

        # Get viewing directions for each ray-box intersection
        viewdirs_box_o = dirs_o[intersection_map[:, 0], intersection_map[:, 1], :]
        viewdirs_box_w = 1 / torch.linalg.norm(rays_d_in_w, dim=1)[:, None] * rays_d_in_w

    else:
        # In case no ray intersects with any object return empty lists
        z_vals_in_w = z_vals_out_w = []
        pts_box_in_w = pts_box_in_o = []
        viewdirs_box_w = viewdirs_box_o = []
        z_ray_out_o = z_ray_in_o = []
    return (
        pts_box_in_w,
        viewdirs_box_w,
        z_vals_in_w,
        z_vals_out_w,
        pts_box_in_o,
        viewdirs_box_o,
        z_ray_in_o,
        z_ray_out_o,
        intersection_map,
        rays_o_o,
    )


def get_closest_intersections(z_vals_w, intersection_map, N_rays, N_obj):
    """Reduces intersections given by z_vals and intersection_map to the first intersection along each ray

    Args:
        z_vals_w: All integration steps for all ray-box intersections in world coordinates [n_intersections,]
        intersection_map: Mapping from flat array to ray-box intersection matrix [n_intersections, 2]
        N_rays: Total number of rays
        N_obj: Total number of objects

    Returns:
        z_vals_w: Integration step for the first ray-box intersection per ray in world coordinates [N_rays,]
        intersection_map: Mapping from flat array to ray-box intersection matrix [N_rays, 2]
        id_first_intersect: Mapping from all intersection related values to first intersection only [N_rays,1]

    """
    # Flat to dense indices
    # Create matching ray-object intersectin matrix with index for all z_vals
    # id_z_vals = tf.scatter_nd(intersection_map, tf.range(z_vals_w.shape[0]), [N_rays, N_obj])
    id_z_vals = (
        torch.zeros((N_rays * N_obj))
        .to(torch.int32)
        .index_add_(
            0,
            intersection_map[:, 0] * N_obj + intersection_map[:, 1],
            torch.arange(0, z_vals_w.shape[0]).to(torch.int32),
        )
        .reshape(N_rays, N_obj)
    )
    # Create ray-index array
    id_ray = torch.arange(0, N_rays).to(torch.int64)
    # id_ray = tf.cast(tf.range(N_rays), tf.int64)

    # Flat to dense values
    # Scatter z_vals in world coordinates to ray-object intersection matrix
    # z_scatterd = tf.scatter_nd(intersection_map, z_vals_w, [N_rays, N_obj])
    z_scatterd = (
        torch.zeros((N_rays * N_obj))
        .to(torch.float32)
        .index_add_(0, intersection_map[:, 0] * N_obj + intersection_map[:, 1], z_vals_w.to(torch.float32))
        .reshape(N_rays, N_obj)
    )

    # Set empty intersections to 1e10
    # z_scatterd_nz = tf.where(tf.equal(z_scatterd, 0), tf.ones_like(z_scatterd) * 1e10, z_scatterd)
    z_scatterd_nz = torch.where(z_scatterd == 0, torch.ones_like(z_scatterd) * 1e10, z_scatterd)

    # Get minimum values along each ray and corresponding ray-box intersection id
    # id_min = tf.argmin(z_scatterd_nz, axis=1)
    id_min = torch.argmin(z_scatterd_nz, dim=1)
    # id_reduced = tf.concat([id_ray[:, tf.newaxis], id_min[:, tf.newaxis]], axis=1)
    id_reduced = torch.cat((id_ray[:, None], id_min[:, None]), dim=1)
    # z_vals_w_reduced = tf.gather_nd(z_scatterd, id_reduced)

    # z_vals_w_reduced = tf.gather_nd(z_scatterd, id_reduced)
    z_vals_w_reduced = z_scatterd[id_reduced[:, 0], id_reduced[:, 1]]

    # Remove all rays w/o intersections (min(z_vals_reduced) == 0)
    # id_non_zeros = tf.where(tf.not_equal(z_vals_w_reduced, 0))
    id_non_zeros = torch.where(z_vals_w_reduced != 0)[0]
    if len(id_non_zeros) != N_rays:
        z_vals_w_reduced = z_vals_w_reduced[id_non_zeros]
        id_reduced = id_reduced[id_non_zeros, :]
        # z_vals_w_reduced = tf.gather_nd(z_vals_w_reduced, id_non_zeros)
        # id_reduced = tf.gather_nd(id_reduced, id_non_zeros)

    # Get intersection map only for closest intersection to the ray origin
    intersection_map_reduced = id_reduced
    # id_first_intersect = tf.gather_nd(id_z_vals, id_reduced)[:, tf.newaxis]
    id_first_intersect = id_z_vals[id_reduced[:, 0], id_reduced[:, 1]].reshape(-1).to(torch.int64)

    return z_vals_w_reduced, intersection_map_reduced, id_first_intersect


def combine_z(z_vals_bckg, z_vals_obj_w, intersection_map, N_rays, N_samples, N_obj, N_samples_obj=1):
    """Combines and sorts background node and all object node intersections along a ray

    Args:
        z_vals_bckg: integration step along each ray [N_rays, N_samples]
        z_vals_obj_w:  integration step of ray-box intersection in the world frame [n_intersects, N_samples_obj
        intersection_map: mapping of points, viewdirs and z_vals to the specific rays and objects at ray-box intersection
        N_rays: Amount of rays
        N_samples: Amount of samples along each ray
        N_obj: Maximum number of objects
        N_samples_obj: Number of samples per object

    Returns:
        z_vals:  [N_rays, N_samples + N_samples_obj*N_obj, 4]
        id_z_vals_bckg:
        id_z_vals_obj:
    """
    if z_vals_obj_w is None or z_vals_obj_w.shape == torch.Size([1]):
        z_vals_obj_w_sparse = torch.zeros([N_rays, N_obj * N_samples_obj])
    else:
        intersection_map = intersection_map[:, 0] * N_obj + intersection_map[:, 1]
        z_vals_obj_w_sparse = (
            torch.zeros([N_rays * N_obj, N_samples_obj])
            .to(intersection_map.device)
            .index_add_(0, intersection_map, z_vals_obj_w)
            .reshape(N_rays, N_obj, N_samples_obj)
        )
        # z_vals_obj_w_sparse = tf.scatter_nd(intersection_map, z_vals_obj_w, [N_rays, N_obj, N_samples_obj])    #TODO
        z_vals_obj_w_sparse = torch.reshape(z_vals_obj_w_sparse, [N_rays, N_samples_obj * N_obj])

    sample_range = torch.arange(0, N_rays).to(intersection_map.device)
    obj_range = (sample_range.unsqueeze(-1).unsqueeze(-1).repeat_interleave(N_obj, dim=1)).repeat_interleave(
        N_samples_obj, dim=2
    )
    # obj_range = tf.repeat(tf.repeat(sample_range[:, tf.newaxis, tf.newaxis], N_obj, axis=1), N_samples_obj, axis=2)

    # Get ids to assign z_vals to each model
    if z_vals_bckg is not None:
        if len(z_vals_bckg.shape) < 2:
            # z_vals_bckg = z_vals_bckg[tf.newaxis]
            z_vals_bckg = z_vals_bckg.unsqueeze()
        # Combine and sort z_vals along each ray
        z_vals, z_vals_indices = torch.sort(torch.cat((z_vals_obj_w_sparse, z_vals_bckg), dim=1), dim=1)

        bckg_range = sample_range.unsqueeze(-1).unsqueeze(-1).repeat_interleave(N_samples, dim=1)
        id_z_vals_bckg = torch.cat(
            (bckg_range, torch.searchsorted(z_vals, z_vals_bckg.contiguous()).unsqueeze(-1)), dim=2
        )
        # bckg_range = tf.repeat(sample_range[:, tf.newaxis, tf.newaxis], N_samples, axis=1)
        # id_z_vals_bckg = tf.concat([bckg_range, tf.searchsorted(z_vals, z_vals_bckg)[..., tf.newaxis]], axis=2)
    else:
        z_vals, z_vals_indices = torch.sort(z_vals_obj_w_sparse, dim=1)
        id_z_vals_bckg = None

    # id_z_vals_obj = tf.concat([obj_range, tf.searchsorted(z_vals, z_vals_obj_w_sparse)], axis=2)
    id_z_vals_obj = torch.cat(
        [
            obj_range.unsqueeze(-1),
            torch.reshape(torch.searchsorted(z_vals, z_vals_obj_w_sparse), (N_rays, N_obj, N_samples_obj)).unsqueeze(
                -1
            ),
        ],
        dim=-1,
    )

    # id_z_vals_obj = tf.concat([obj_range[..., tf.newaxis],
    #                            tf.reshape(tf.searchsorted(z_vals, z_vals_obj_w_sparse), [N_rays, N_obj, N_samples_obj])[..., tf.newaxis]
    #                            ], axis=-1)

    return z_vals, id_z_vals_bckg, id_z_vals_obj


# def render_mot_scene(pts, viewdirs, network_fn, network_query_fn,
#                      inputs, viewdirs_obj, z_vals_in_o, n_intersect, object_idx, object_y, obj_pose,
#                      unique_classes, class_id, latent_vector_dict, object_network_fn_dict,
#                      N_rays,N_samples, N_obj, N_samples_obj,
#                      obj_only=False):
#
#     # Prepare raw output array
#     raw = tf.zeros([N_rays, N_samples + N_obj * N_samples_obj, 4]) if not obj_only else tf.zeros([N_rays, N_obj * N_samples_obj, 4])
#     raw_sh = raw.shape
#
#     if not obj_only:
#         # Predict RGB and density from background
#         raw_bckg = network_query_fn(pts, viewdirs, network_fn)
#         raw += tf.scatter_nd(id_z_vals_bckg, raw_bckg, raw_sh)
#
#     # Check for object intersections
#     if z_vals_in_o is not None:
#         # Loop for one model per object and no latent representations
#         if latent_vector_dict is None:
#             obj_id = tf.reshape(object_idx, obj_pose[..., 4].shape)
#             for k, track_id in enumerate(object_y):
#                 if track_id >= 0:
#                     input_indices = tf.where(tf.equal(obj_id, k))
#                     input_indices = tf.reshape(input_indices, [-1, N_samples_obj, 2])
#                     model_name = 'model_obj_' + str(np.array(track_id).astype(np.int32))
#                     # print('Hit', model_name, n_intersect, 'times.')
#                     if model_name in object_network_fn_dict:
#                         obj_network_fn = object_network_fn_dict[model_name]
#
#                         inputs_obj_k = tf.gather_nd(inputs, input_indices)
#                         viewdirs_obj_k = tf.gather_nd(viewdirs_obj,
#                                                       input_indices[..., None, 0]) if N_samples_obj == 1 else \
#                             tf.gather_nd(viewdirs_obj, input_indices[..., None, 0, 0])
#
#                         # Predict RGB and density from object model
#                         raw_k = network_query_fn(inputs_obj_k, viewdirs_obj_k, obj_network_fn)
#
#                         if n_intersect is not None:
#                             # Arrange RGB and denisty from object models along the respective rays
#                             raw_k = tf.scatter_nd(input_indices[:, :], raw_k, [n_intersect, N_samples_obj,
#                                                                                4])  # Project the network outputs to the corresponding ray
#                             raw_k = tf.scatter_nd(intersection_map[:, :2], raw_k, [N_rays, N_obj, N_samples_obj,
#                                                                                    4])  # Project to rays and object intersection order
#                             raw_k = tf.scatter_nd(id_z_vals_obj, raw_k, raw_sh)  # Reorder along z and ray
#                         else:
#                             raw_k = tf.scatter_nd(input_indices[:, 0][..., tf.newaxis], raw_k, [N_rays, N_samples, 4])
#
#                         # Add RGB and density from object model to the background and other object predictions
#                         raw += raw_k
#         # Loop over classes c and evaluate each models f_c for all latent object describtor
#         else:
#             for c, class_type in enumerate(unique_classes.y):
#                 # Ignore background class
#                 if class_type >= 0:
#                     input_indices = tf.where(tf.equal(class_id, c))
#                     input_indices = tf.reshape(input_indices, [-1, N_samples_obj, 2])
#                     model_name = 'model_class_' + str(int(np.array(class_type))).zfill(5)
#
#                     if model_name in object_network_fn_dict:
#                         obj_network_fn = object_network_fn_dict[model_name]
#
#                         inputs_obj_c = tf.gather_nd(inputs, input_indices)
#
#                         # Legacy version 2
#                         # latent_vector = tf.concat([
#                         #         latent_vector_dict['latent_vector_' + str(int(obj_id)).zfill(5)][tf.newaxis, :]
#                         #         for obj_id in np.array(tf.gather_nd(obj_pose[..., 4], input_indices)).astype(np.int32).flatten()],
#                         #         axis=0)
#                         # latent_vector = tf.reshape(latent_vector, [inputs_obj_k.shape[0], inputs_obj_k.shape[1], -1])
#                         # inputs_obj_k = tf.concat([inputs_obj_k, latent_vector], axis=-1)
#
#                         # viewdirs_obj_k = tf.gather_nd(viewdirs_obj,
#                         #                               input_indices[..., 0]) if N_samples_obj == 1 else \
#                         #     tf.gather_nd(viewdirs_obj, input_indices)
#
#                         viewdirs_obj_c = tf.gather_nd(viewdirs_obj, input_indices[..., None, 0])[:, 0, :]
#
#                         # Predict RGB and density from object model
#                         raw_k = network_query_fn(inputs_obj_c, viewdirs_obj_c, obj_network_fn)
#
#                         if n_intersect is not None:
#                             # Arrange RGB and denisty from object models along the respective rays
#                             raw_k = tf.scatter_nd(input_indices[:, :], raw_k, [n_intersect, N_samples_obj,
#                                                                                4])  # Project the network outputs to the corresponding ray
#                             raw_k = tf.scatter_nd(intersection_map[:, :2], raw_k, [N_rays, N_obj, N_samples_obj,
#                                                                                    4])  # Project to rays and object intersection order
#                             raw_k = tf.scatter_nd(id_z_vals_obj, raw_k,
#                                                   raw_sh)  # Reorder along z in  positive ray direction
#                         else:
#                             raw_k = tf.scatter_nd(input_indices[:, 0][..., tf.newaxis], raw_k,
#                                                   [N_rays, N_samples, 4])
#
#                         # Add RGB and density from object model to the background and other object predictions
#                         raw += raw_k
#                     else:
#                         print('No model ', model_name, ' found')
#
#     return raw
