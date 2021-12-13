#!/usr/bin/env python3.7
import torch
import numpy as np
import cv2

class Dataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self,list_IDs, labels):

        self.labels = labels
        self.list_IDs = list_IDs
        self.Im = np.zeros((1,128,128))
        self.X_matrix = np.zeros((50000,7))
        # Opening files
        x_file = open('JointStates_q.txt', 'r+')
        # Reading from the file
        content_x = x_file.readlines()
        # Definition of the matrices
        x_matrix = []
        dim = 50000
        for line in content_x:
            x_matrix = np.matrix(line, float)

        c=0
        for i in range(dim):
           for l in range(7):
               self.X_matrix[i,l] = x_matrix[0,c]
               c=c+1


        self.adjustments = np.array([2.5, 2.5, 1.3, 4.2, 1.3, 1.2, 2.5], float)


    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, idx):

      
       img_path = 'Dataset/camera_image_' + str(idx) + '.jpeg'
       image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
       self.Im[0,:,:] = image.astype("float32")/255
       joints = self.X_matrix[idx, :] + self.adjustments

       return joints, self.Im




