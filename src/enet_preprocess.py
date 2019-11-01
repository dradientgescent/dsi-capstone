
import os
import sys
# Repository source: https://github.com/qubvel/efficientnet
sys.path.append(os.path.abspath('../../efficientnet/'))
from efficientnet.model import EfficientNetB5
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

train_path = '/media/brats/mirlproject2/aptos_2019/train/'
test_path = '/media/brats/mirlproject2/aptos_2019/test/'
savepath_train = '/media/brats/mirlproject2/aptos_2019/train_cropped/'
savepath_test = '/media/brats/mirlproject2/aptos_2019/test_cropped/'

def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the orignal image and 
    returns the a preprocessed image with 
    3 channels
    """
    # If for some reason we only have two channels
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_image(image, sigmaX=10, resize=(256,256)):
    """
    The whole preprocessing pipeline:
    1. Read in image
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = crop_image_from_gray(image)
    image = cv2.resize(image, resize)
    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
    return image

def crop_images(path, new_path):

	os.makedirs(new_path, exist_ok = True)

	dirs = [l for l in os.listdir(path) if l != '.DS_Store']
	total = 0

	for item in dirs:
		if not os.path.exists(new_path+item):
			try:
				print(item)
				img = cv2.imread(path+item)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				# plt.imshow(img)
				# plt.show()
				img = crop_image_from_gray(img)

				cv2.imwrite(str(new_path + item), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
				total += 1
				print("Saving: ", item, total)
			except Exception as e:
				print(e)


# crop_images(train_path, savepath_train)
# crop_images(test_path, savepath_test)
# img = cv2.imread('/media/brats/mirlproject2/aptos_2019/train_cropped/0c917c372572.png')
# print(np.ptp(img))
# plt.imshow(preprocess_image(cv2.imread('/media/brats/mirlproject2/aptos_2019/train_cropped/0c917c372572.png')))
# plt.show()
# im = np.asarray(Image.open(r"/media/brats/mirlproject2/aptos_2019/train_cropped/0c917c372572.png"))
# print(np.ptp(im))