#! /usr/bin/python2.7
# rospy for the subscriber
#import rospy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

class Im_Process:

     def __init__(self):
        
        np.set_printoptions(threshold=sys.maxsize)

        self.Im = np.zeros((1, 128, 128))
        self.X_matrix = np.zeros((50000, 7))
        # Opening files
        x_file = open('JointStates_q.txt', 'r+')
        # Reading from the file
        content_x = x_file.readlines()
        # Definition of the matrices
        x_matrix = []
        dim = 50000
        for line in content_x:
            x_matrix = np.matrix(line, float)

        c = 0
        for i in range(dim):
            for l in range(7):
                self.X_matrix[i, l] = x_matrix[0, c]
                c = c + 1

        Y = np.zeros((50000, 128, 128))
        dim = 50000
       
        for i in range(dim):
            img_path = 'Dataset/camera_image_' + str(i) + '.jpeg'
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            Y[i, :, :] = image.astype("float32")/255

        self.im_std = np.std(Y, 0)
        print(self.im_std)
        self.joint_min = np.min(self.X_matrix, 0)
        print(self.joint_min)
        self.joint_max = np.max(self.X_matrix, 0)
        print(self.joint_max)
        self.joint_std = np.std(self.X_matrix, 0)
        print(self.joint_std)
        np.savetxt("Im_STD.csv", self.im_std, delimiter=',')
        np.savetxt("q_min.csv", self.joint_min, delimiter=',')
        np.savetxt("q_max.csv", self.joint_max, delimiter=',')
        np.savetxt("q_STD.csv", self.joint_std, delimiter=',')



if __name__ == '__main__':

      p = Im_Process()

   
