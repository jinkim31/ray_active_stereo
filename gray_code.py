import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def decode(graycode, images, valid_threshold=50):
    _, patterns = graycode.generate()
    assert (len(patterns)+2 == len(images))
    bit_depth = (len(patterns))//4
    projector_resolution = patterns[0].shape # TODO: Filter out erroneous decoding results exceeding projector resolution
    image_resolution = images[0].shape

    valid_mask = (images[1].astype(np.int16) - images[0].astype(np.int16)) > valid_threshold
    images = images[2:]

    decoded_x = np.zeros(image_resolution, dtype=np.uint32)
    decoded_y = np.zeros(image_resolution, dtype=np.uint32)
    xor_x = np.zeros(image_resolution, dtype=np.uint8)
    xor_y = np.zeros(image_resolution, dtype=np.uint8)

    for bit in range(bit_depth):
        binary_x = (images[2*bit+0] > images[2*bit+1]).astype(np.uint8)
        binary_y = (images[2*bit+0+len(images)//2] > images[2*bit+1+len(images)//2]).astype(np.uint8)
        xor_x = xor_x ^ binary_x
        xor_y = xor_y ^ binary_y
        multiplier = 2 ** (bit_depth - 1 - bit)
        decoded_x += xor_x.astype(np.uint32) * multiplier
        decoded_y += xor_y.astype(np.uint32) * multiplier

    decoded_x[~valid_mask] = 0
    decoded_y[~valid_mask] = 0

    return decoded_x, decoded_y


def stereo_match(decoded_l, decoded_r):
    """
    :param decoded_l:
    :param decoded_r:
    :return: Array of image L points (x, y), array of image R points (x, y)
    """
    ys_l, xs_l = np.where((decoded_l[0] > 0) & (decoded_l[1] > 0))
    image_l_points = [(y, x) for y, x in zip(ys_l, xs_l)] # array of points (y, x)
    pattern_points = [(decoded_l[1][*point], decoded_l[0][*point]) for point in image_l_points]
    transform_pattern_to_image_r = {} # (pattern_y, pattern_x):(image_r_y, image_r_x) dictionary
    ys_r, xs_r = np.where((decoded_r[0] > 0) & (decoded_r[1] > 0))
    for y, x in zip(ys_r, xs_r):
        transform_pattern_to_image_r[decoded_r[1][y, x], decoded_r[0][y, x]] = (y, x)
    image_r_points = [transform_pattern_to_image_r.get(point) for point in pattern_points]
    image_l_points, image_r_points = zip(*[(x, y) for x, y in zip(image_l_points, image_r_points) if x is not None and y is not None])
    #return np.array(image_l_points), np.array(image_r_points)
    return np.array(image_l_points)[:, ::-1], np.array(image_r_points)[:, ::-1]
