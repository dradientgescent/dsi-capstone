import os
import sys
# from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from skimage.transform import resize
import numpy as np
import cv2
from glob import glob


def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.

    INPUT
        directory: Folder to be created, called as "folder/".

    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def crop_and_resize_images(path, new_path, cropx, cropy, img_size=256):
    '''
    Crops, resizes, and stores all images from a directory in a new directory.

    INPUT
        path: Path where the current, unscaled images are contained.
        new_path: Path to save the resized images.
        img_size: New size for the rescaled images.

    OUTPUT
        All images cropped, resized, and saved from the old folder to the new folder.
    '''
    create_directory(new_path)
    dirs = [l for l in os.listdir(path) if l != '.DS_Store']
    total = 0

    for item in dirs:
        if not os.path.exists(new_path+item):
            try:
                print(item)
                img = cv2.imread(path+item)
                _,thresh1 = cv2.threshold(img,25,255,cv2.THRESH_BINARY)

                x, y = np.where(thresh1)[0], np.where(thresh1)[1]
                xmin, ymin = min(x), min(y)
                xmax, ymax = max(x), max(y)
                # y,x,channel = img.shape
                # startx = x//2-(cropx//2)
                # starty = y//2-(cropy//2)
                img = img[xmin:xmax, ymin:ymax]
                #img = cv2.resize(img, (512,512))
                cv2.imwrite(str(new_path + item), img)
                total += 1
                print("Saving: ", item, total)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    crop_and_resize_images(path='/media/bmi/poseidon/DiabeticR/train/', new_path='/media/bmi/poseidon/DiabeticR/train_cropped/', cropx=1800, cropy=1800, img_size=256)
    #crop_and_resize_images(path='/media/parth/DATA/DiabeticR/val/', new_path='/media/parth/DATA/DiabeticR/val_resized/', cropx=1800, cropy=1800, img_size=256)
