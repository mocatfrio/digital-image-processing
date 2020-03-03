import sys
import os
from math import ceil
import cv2
import matplotlib.pyplot as plt

import constant
from os_handler import get_list, check_dir

def generate_plot(total_img, num_img_each_row=3):
    row = ceil(total_img / num_img_each_row) * 100
    col = num_img_each_row * 10
    return row + col


def display_all_img(src_path, dest_path, img_name):
    imgs_name = [img_name] + get_list(dest_path)
    # generate plot
    plot = generate_plot(len(imgs_name))
    # plotting
    for i in range(plot, plot + len(imgs_name)):
        title = imgs_name[i - plot].split('_')[0]
        img_path = src_path + imgs_name[i - plot] if i == plot else dest_path + imgs_name[i - plot]
        img = cv2.imread(img_path)[:,:,::-1]
        plt.subplot(i + 1), plt.imshow(img)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
    # display
    # plt.show()
    all_img_path = src_path + 'all/'
    check_dir(all_img_path)
    plt.savefig(all_img_path + img_name.split('.')[0] + '.png')


def display_histogram(src_path, dest_path, img_name, is_grayscale=True):
    imgs_name = [img_name] + get_list(dest_path)
    # generate plot
    plot = generate_plot(len(imgs_name))
    # plotting
    for i in range(plot, plot + len(imgs_name)):
        title = imgs_name[i - plot].split('_')[0]
        img_path = src_path + imgs_name[i - plot] if i == plot else dest_path + imgs_name[i - plot]
        if is_grayscale:
            img = cv2.imread(img_path, 0)
            plt.subplot(i + 1), plt.hist(img.ravel(),256,[0,256])
            plt.title(title)
            plt.xticks([]), plt.yticks([])
        else:
            img = cv2.imread(img_path)
            color = ('b','g','r')
            plt.subplot(i + 1)
            for i, col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,256])
                plt.plot(histr, color = col)
                plt.xlim([0,256])
            plt.title(title)
            plt.xticks([]), plt.yticks([])
    # display
    # plt.show()
    hist_path = src_path + 'gray_hist/' if is_grayscale else src_path + 'rgb_hist/'
    check_dir(hist_path)
    plt.savefig(hist_path + img_name.split('.')[0] + '.png')


if __name__ == "__main__":
    command = sys.argv[1]
    train_imgs = get_list(constant.SRC_IMG_PATH)
    for img_name in train_imgs:
        src_path = constant.SRC_IMG_PATH
        dest_path = constant.DEST_IMG_PATH + img_name.split('.')[0] + '/'
        if command == 'img':
            display_all_img(src_path, dest_path, img_name)
        elif command == 'hist':
            opt_command = sys.argv[2]
            if opt_command == 'gray':
                display_histogram(src_path, dest_path, img_name)
            else:
                display_histogram(src_path, dest_path, img_name, False)
