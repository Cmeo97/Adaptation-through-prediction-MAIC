#!/usr/bin/env python3.7
import torch
import numpy as np
import cv2
from Images_normalizer import Im_Process

def main():

   i = 2
   for idx in range(50000):
      img_path = 'Dataset/camera_image_' + str(idx) + '.jpeg'
      image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      if (image is None):
         filename = 'Dataset/camera_image_' + str(idx) + '.jpeg'
         img_path = 'Dataset/camera_image_' + str(i) + '.jpeg'
         image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
         cv2.imwrite(filename, image)
         print(idx)
      else:
         i = idx

if __name__ == '__main__':
      main()
