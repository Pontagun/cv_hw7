import cv2
import numpy as np
from PIL import Image
import math


import matplotlib.pyplot as plt
from numpy import ceil
from numpy.distutils.misc_util import get_cmd

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float


def get_alpha(row, col):
    return np.rad2deg(np.arctan(row / col))


def get_logr(row, col):
    return np.log(np.sqrt((col ** 2) + (row ** 2)))


def pixel_log255(unorm_image):
    pxmin = unorm_image.min()
    pxmax = unorm_image.max()

    for i in range(unorm_image.shape[0]):
        for j in range(unorm_image.shape[1]):
            unorm_image[i, j] = (255 / math.log10(256)) * math.log10(1 + (255 / pxmax) * unorm_image[i, j])
            # unorm_image[i, j] = ((unorm_image[i, j] - pxmin) / (pxmax - pxmin)) * 255

    norm_image = unorm_image
    return norm_image

if __name__ == '__main__':
    img_path = r"Capture - Copy (2).PNG"
    image = cv2.imread(img_path, 0)

    origi_height, origi_width = image.shape
    center_height, center_width = int(origi_height / 2), int(origi_width / 2)
    offset_r = center_height - 1
    offset_c = center_width - 1

    log_image = np.ones((origi_height, origi_width))

    prev_alpha = 0
    for row in range(center_height, origi_height, 1):
        for col in range(center_width, origi_width, 1):
            temp_col = offset_c - col
            temp_row = offset_r - row

            alpha = int(get_alpha(temp_row, temp_col))
            logr = int(get_logr(temp_row, temp_col) * 40)

            log_image[prev_alpha + alpha][logr] = image[row][col]

    prev_alpha = 175
    for row in range(center_height, origi_height, 1):
        for col in range(center_width, 0, -1):
            temp_col = offset_c + 2 - col
            temp_row = offset_r - row

            alpha = int(get_alpha(temp_row, temp_col))
            logr = int(get_logr(temp_row, temp_col) * 40)

            log_image[prev_alpha + alpha][logr] = image[row][col]

    prev_alpha = 354
    for row in range(center_height, -1, -1):
        for col in range(center_width, origi_width, 1):
            temp_col = offset_c - col
            temp_row = offset_r - row

            alpha = int(get_alpha(temp_row, temp_col))
            logr = int(get_logr(temp_row, temp_col) * 40)

            log_image[prev_alpha + alpha][logr] = image[row][col]

    prev_alpha = 175

    for row in range(center_height, -1, -1):
        for col in range(center_width, 0, -1):
            temp_col = offset_c + 2 - col
            temp_row = offset_r - row

            alpha = int(get_alpha(temp_row, temp_col))
            logr = int(get_logr(temp_row, temp_col) * 40)

            log_image[prev_alpha + alpha][logr] = image[row][col]


    img = Image.fromarray(log_image)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img.save("logr_imglg-2.jpg")
    #
    image_polar = warp_polar(image, radius=origi_width)
    plt.imshow(image_polar)
    plt.show()



