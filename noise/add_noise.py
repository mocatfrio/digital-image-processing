import random
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import constant
from os_handler import get_list, check_dir

def apply_all_noise(img):
    noise_imgs = {}
    noise = {
        'gaussian': apply_gaussian,
        'salt-pepper': apply_salt_pepper,
        'salt': apply_salt_pepper,
        'pepper': apply_salt_pepper,
        'poisson': apply_poisson,
        'speckle': apply_speckle
    }
    for key, value in noise.items():
        if key == 'salt':
            noise_img = value(img, True, False)
        elif key == 'pepper':
            noise_img = value(img, False, True)
        else:
            noise_img = value(img)
        noise_imgs[key] = noise_img
    return noise_imgs

def apply_gaussian(img):
    output = cv2.GaussianBlur(img, (5,5), 0)
    return output
        
def apply_salt_pepper(img, is_salt=True, is_pepper=True, prob=0.05):
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rand = random.random()
            if rand < prob:
                output[i][j] = 0 if is_pepper else img[i][j]
            elif rand > thres:
                output[i][j] = 255 if is_salt else img[i][j]
            else:
                output[i][j] = img[i][j]
    return output
            
def apply_poisson(img):
    noise = np.random.poisson(50, img.shape)
    output = img + noise
    return output

def apply_speckle(img, prob=0.07):
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rand = random.random()
            if rand < prob:
                output[i][j] = 128
                for k in range(5):
                    output[i-k][j-k] = 128 + 10*rand
            else:
                output[i][j] = img[i][j]
    return output

if __name__ == "__main__":
    # get all train images
    train_imgs = get_list(constant.SRC_IMG_PATH)
    for img_name in train_imgs:
        src_path = constant.SRC_IMG_PATH + img_name
        dest_path = constant.DEST_IMG_PATH + img_name.split('.')[0] + '/'
        check_dir(dest_path)
        # read original image
        img = cv2.imread(src_path)[:,:,::-1]
        # apply noise filter
        noise_imgs = apply_all_noise(img)
        # export image
        for key, value in noise_imgs.items():
            cv2.imwrite(dest_path + key + '_' + img_name, value[:,:,::-1])
