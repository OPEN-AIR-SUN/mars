import cv2
import os
import numpy as np
import argparse

parsers = argparse.ArgumentParser(description="KITTI-STEP to panoptic segmentation")
parsers.add_argument("--annotation_path", type=str, default="")
parsers.add_argument("--output_path", type=str, default="")
annotation_path, output_path = parsers.parse_args().annotation_path, parsers.parse_args().output_path

colormap = np.zeros((256, 3), dtype=np.uint8)

colormap[0] = [128, 64, 128]
colormap[1] = [244, 35, 232]
colormap[2] = [70, 70, 70]
colormap[3] = [102, 102, 156]
colormap[4] = [190, 153, 153]
colormap[5] = [153, 153, 153]
colormap[6] = [70, 130, 180]
colormap[7] = [220, 220, 0]
colormap[8] = [107, 142, 35]
colormap[9] = [152, 251, 152]
colormap[10] = [250, 170, 30]
colormap[11] = [220, 20, 60]
colormap[12] = [255, 0, 0]
colormap[13] = [0, 0, 142]
colormap[14] = [0, 0, 70]
colormap[15] = [0, 60, 100]
colormap[16] = [0, 80, 100]
colormap[17] = [0, 0, 230]
colormap[18] = [119, 11, 32]
colormap[255] = [0, 0, 0]

for filename in os.listdir(annotation_path):
    file_path = os.path.join(annotation_path, filename)
    img = cv2.imread(file_path)
    modified_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i, j]
            r, g, b = pixel[0], pixel[1], pixel[2]
            last_value = b
            modified_img[i, j] = colormap[b]

    modified_file_path = os.path.join(output_path, filename)
    cv2.imwrite(modified_file_path, modified_img)
