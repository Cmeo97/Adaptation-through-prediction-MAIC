#!/usr/bin/env python2.7
import argparse
from MAIC_full import *
import gc
import GPUtil
import time
import subprocess
import scipy.io
from multiprocessing import Process
import sys


def main():

    rospy.init_node('AIC_controller')
    AIC = AIC_agent()

    #Desired Poses
    Joints_data = scipy.io.loadmat('Joints_1000.mat')
    Joints_desired = Joints_data['Joints']
 
    desiredPos1 = Joints_desired[:, 500]
    desiredPos2 = Joints_desired[:, 300]
    desiredPos3 = Joints_desired[:, 560]
    desiredPos4 = Joints_desired[:, 110]
    desiredPos5 = Joints_desired[:, 700]


    im_path_1 = 'goal_poses/image_goal_1.jpeg'
    im_path_2 = 'goal_poses/image_goal_2.jpeg'
    im_path_3 = 'goal_poses/image_goal_3.jpeg'
    im_path_4 = 'goal_poses/image_goal_4.jpeg'
    im_path_5 = 'goal_poses/image_goal_5.jpeg'
    

    AIC.setGoal(desiredPos5, 5)
    #AIC.end_effector_selection(0)
    rate = rospy.Rate(165)
    count = 0
    while not rospy.is_shutdown():

        if count == 0:
            AIC.get_latent_action()

        if count > 0:
            p2 = Process(target=AIC.minimiseF())

        count += 1

        if count == 800:
            AIC.setGoal(desiredPos1, im_path_1)

        if count == 1600:
            AIC.setGoal(desiredPos2, im_path_2)

        if count == 2400:
            AIC.setGoal(desiredPos3, im_path_3)

        if count == 3200:
            AIC.setGoal(desiredPos4, im_path_4)

        if count == 4000:
            AIC.setGoal(desiredPos5, im_path_5)


        if count > 4799:
            AIC.logger()
            sys.exit()

    rate.sleep()


if __name__ == "__main__":
    main()
