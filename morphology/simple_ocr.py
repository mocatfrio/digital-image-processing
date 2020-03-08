# import libraries
from PIL import Image
import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import os


def ocr(image, key):
    RESULT_PATH = "result/"
    filename = key + '_output.txt'
    check_dir(RESULT_PATH)
    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    with open(RESULT_PATH + filename, 'w') as f:
        f.write(pytesseract.image_to_string(image, config=custom_config))


def check_dir(path):
    # make directory
    if not os.path.exists(path):
        os.makedirs(path)

# Preprocessing
# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal - using median filter
def remove_noise(image, kernel=5):
    return cv2.medianBlur(image, kernel)
 
# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# dilation
def dilate(image, kernel=np.ones((5,5),np.uint8)):
    return cv2.dilate(image, kernel, iterations = 1)
    
# erosion
def erode(image, kernel=np.ones((5,5),np.uint8)):
    return cv2.erode(image, kernel, iterations = 1)

# opening - erosion followed by dilation
def opening(image, kernel=np.ones((5,5),np.uint8)):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image, kernel=np.ones((5,5),np.uint8)):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

def testing(image):
    DEST_PATH = 'prep_img/'
    check_dir(DEST_PATH)

    # ocr(image, 'original')
    gray_image = get_grayscale(image)
    scenarios = {
        'thresholding': thresholding,
        'opening': opening,
        'closing': closing,
        'dilate': dilate,
        'erode': erode,
        'canny': canny
    }
    for key, function in scenarios.items():
        prep_image = function(gray_image)
        # ocr(prep_image, key)
        filename = key + '_img.jpg'
        try:
            cv2.imwrite(DEST_PATH + filename, prep_image[:,:,::-1])
        except IndexError:
            cv2.imwrite(DEST_PATH + filename, prep_image)


if __name__ == "__main__":
    IMG_PATH = "img/"

    image = cv2.imread(IMG_PATH + "metpen.jpg")[:,:,::-1]
    testing(image)

    display_all_preprocessed_img()
