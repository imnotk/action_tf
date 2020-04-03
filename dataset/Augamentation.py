from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
from PIL import ImageEnhance,Image
import random

class DataAugmentation:

    def __init__(self, image = None):
        if image:
            self.image = image
            self.height = image.shape[0]
            self.width = image.shape[1]


    @staticmethod
    def horizontal_flip(image,is_flow=False):
        k = random.random()
        if k < 0.5:
            if len(image.shape) == 3:
                    return np.fliplr(image)
            else:
                    return image[:,:,::-1,:]
        else:
            return image

    @staticmethod
    def Multiscale_crop(image,is_flow=False):

        if is_flow:
            random_size = [256,224,192]
        else:
            random_size = [256, 224, 192, 168]

        crop_list = []
        for i in random_size:
            for j in random_size:
                crop_list.append((i,j))
        assert len(image.shape) ==  3 or len(image.shape) == 4

        if len(image.shape) == 3:
            height = image.shape[0]
            width = image.shape[1]
        elif len(image.shape) == 4:
            height = image.shape[1]
            width = image.shape[2]

        crop_size = random.choice(crop_list)
        assert 256 in crop_size or 224 in crop_size or 192 in crop_size or 168 in crop_size

        random_crop_location = np.random.randint(0,5)
            
        if random_crop_location == 0:  # left top
            if len (image.shape) == 3:
                return image[0:crop_size[0], 0:crop_size[1]]
            elif len (image.shape) == 4:
                return image[:, 0:crop_size[0], 0:crop_size[1], :]
        elif random_crop_location == 1:  # right top
            if len (image.shape) == 3:
                return image[0:crop_size[0], width - crop_size[1]:width]
            elif len (image.shape) == 4:
                return image[:, 0:crop_size[0], width - crop_size[1]:width, :]
        elif random_crop_location == 2:  # left bottom
            if len (image.shape) == 3:
                return image[height - crop_size[0]:height, 0:crop_size[1]]
            elif len (image.shape) == 4:
                return image[:, height - crop_size[0]:height, 0:crop_size[1], :]
        elif random_crop_location == 3:  # right bottom
            if len (image.shape) == 3:
                return image[height - crop_size[0]:height, width - crop_size[1]:width]
            elif len (image.shape) == 4:
                return image[:, height - crop_size[0]:height,
                       width - crop_size[1]:width, :]
        elif random_crop_location == 4:  # center
            if len (image.shape) == 3:
                return image[
                       (height - crop_size[0]) >> 1: (height + crop_size[0]) >> 1,
                       (width - crop_size[1]) >> 1:(width + crop_size[1]) >> 1]
            elif len (image.shape) == 4:
                return image[:,
                       (height - crop_size[0]) >> 1: (height + crop_size[0]) >> 1,
                       (width - crop_size[1]) >> 1:(width + crop_size[1]) >> 1, :]


    @staticmethod
    def random_Crop(image, random_crop_size=None,random_crop_location = None):
        random_size = [256, 224, 192, 168]
        if random_crop_size is None:
            random_crop_size = np.random.randint(0, 4)
        if random_crop_location is None:
            random_crop_location = np.random.randint(0,5)

        if len(image.shape) == 3:
            height = image.shape[0]
            width = image.shape[1]
        else:
            height = image.shape[1]
            width = image.shape[2]

        if random_crop_location == 0:# left top
            if len (image.shape) == 3:
                return image[0:random_size[random_crop_size], 0:random_size[random_crop_size]]
            elif len (image.shape) == 4:
                return image[:,0:random_size[random_crop_size], 0:random_size[random_crop_size],:]
        elif random_crop_location == 1:#right top
            if len (image.shape) == 3:
                return image[0:random_size[random_crop_size], width -random_size[random_crop_size]:width]
            elif len (image.shape) == 4:
                return image[:,0:random_size[random_crop_size], width - random_size[random_crop_size]:width,:]
        elif random_crop_location == 2:#left bottom
            if len (image.shape) == 3:
                return image[height - random_size[random_crop_size]:height, 0:random_size[random_crop_size]]
            elif len (image.shape) == 4:
                return image[:,height - random_size[random_crop_size]:height, 0:random_size[random_crop_size],:]
        elif random_crop_location == 3:#right bottom
            if len (image.shape) == 3:
                return image[height - random_size[random_crop_size]:height, width - random_size[random_crop_size]:width]
            elif len (image.shape) == 4:
                return image[:,height - random_size[random_crop_size]:height, width - random_size[random_crop_size]:width,:]
        elif random_crop_location == 4:#center
            if len (image.shape) == 3:
                return image[(height - random_size[random_crop_size]) >> 1: (height +random_size[random_crop_size]) >> 1,
                       (width - random_size[random_crop_size]) >> 1:(width + random_size[random_crop_size]) >> 1]
            elif len (image.shape) == 4:
                return image[:,
                       (height - random_size[random_crop_size]) >> 1: (height + random_size[random_crop_size]) >> 1,
                       (width - random_size[random_crop_size]) >> 1:(width + random_size[random_crop_size]) >> 1,:]

    @staticmethod
    def randomCrop(image,height,width):
        if len(image.shape) == 3:
            _height = image.shape[0]
            _width = image.shape[1]
        else:
            _height = image.shape[1]
            _width = image.shape[2]
        assert (_height >= height) and (_width >= width)
        start_idx = np.random.randint(0,_height - height)
        start_idy = np.random.randint(0,_width - width)
        if len (image.shape) == 3:
            return image[start_idx:start_idx + height , start_idy : start_idy + width]
        elif len(image.shape) == 4:
            return image[:,start_idx:start_idx + height , start_idy : start_idy + width,:]


    @staticmethod
    def center_crop(image,height,width):
        if len(image.shape) == 3:
            _height = image.shape[0]
            _width = image.shape[1]
        else:
            _height = image.shape[1]
            _width = image.shape[2]

        if len (image.shape) == 3:
            return image[(_height - height) >> 1: (_height + height) >> 1,
                       (_width - width) >> 1:(_width + width) >> 1]
        elif len(image.shape) == 4:
            return image[:,(_height - height) >> 1: (_height + height) >> 1,
                   (_width - width) >> 1:(_width + width) >> 1,:]

    

    @staticmethod
    def borders25(image):
        if len(image.shape) == 3:
            height = image.shape[0]
            width = image.shape[1]
        else:
            height = image.shape[1]
            width = image.shape[2]
        
        crop_size = (256,224,192)
        crop_list = []
        for i in crop_size:
            for j in crop_size:
                crop_list.append((i,j))
        crop_tuple = random.choice(crop_list) 

        _x = np.random.randint(0, 0.25 * height)
        _y = np.random.randint(0, 0.25 * width)
        _x = min(_x,height-crop_tuple[0])
        _y = min(_y,width-crop_tuple[1])
        
        if len(image.shape) == 3:
            return image[_x:_x+crop_tuple[0],_y:_y+crop_tuple[1]]
        elif len(image.shape) == 4:
            return image[:,_x:_x+crop_tuple[0],_y:_y+crop_tuple[1],:]

    @staticmethod
    def normalize(img,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        img = img/255
        img_channel = img.shape[-1]
        mean = mean * (img_channel // len(mean))
        std = std * (img_channel // len(std))
        for i in range(img_channel):
            img[...,i] = (img[...,i] - mean[i]) / std[i]
        return img

    @staticmethod
    def subtract_mean(img,is_rgb=True):
        # if is_rgb:
        mean = [123.68,116.78,103.94]
        for i in range(3):
            img[...,i] = img[...,i] - mean[i] 
        # else:
        #     img -= 114.8
        return img
    
if __name__ == '__main__':
    a = cv2.imread(r'E:\UCF101\UCF101\UCF-101\ApplyEyeMakeup\v_ApplyEyeMakeup_g01_c01\00001.jpg')
    # cv2.imshow('origin',a)
    # cv2.waitKey(0)
    b = DataAugmentation.random_Crop(a,1,2)
    cv2.imshow('jitter',b)
    cv2.waitKey(0)