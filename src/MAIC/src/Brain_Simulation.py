#!/usr/bin/env python2.7
from AIC_full import *
import sys

robot = 1


def main():

    rospy.init_node('AIC_perception')

    AIC = AIC_agent()
    rate = rospy.Rate(1500)
    
    count = 1
    

    Joints_data = scipy.io.loadmat('Joints_1000.mat')
    Joints_desired = Joints_data['Joints']
 
    desiredPos = Joints_desired[:, 25]
    desiredPos1 = Joints_desired[:, 500]
    desiredPos2 = Joints_desired[:, 300]
    desiredPos3 = Joints_desired[:, 560]
    desiredPos4 = Joints_desired[:, 110]
    desiredPos5 = Joints_desired[:, 700]


    im_path_0 = 'goal_data/camera_image_3.jpeg'
    im_path_1 = 'goal_data/camera_image_156.jpeg'
    im_path_2 = 'goal_data/camera_image_270.jpeg'
    im_path_3 = 'goal_data/camera_image_430.jpeg'
    im_path_4 = 'goal_data/camera_image_586.jpeg'
    im_path_5 = 'goal_data/camera_image_818.jpeg'

   
    AIC.setGoal(desiredPos, im_path_0)
    AIC.get_latent_perception(desiredPos2, im_path_2)
    while not rospy.is_shutdown():
        AIC.perception(count)
        if count == 400:
            AIC.setGoal(desiredPos1, im_path_1)
            AIC.end_effector_selection(1)

        if count == 800:
            AIC.setGoal(desiredPos2, im_path_2)
            AIC.end_effector_selection(2)

        if count == 1200:
            AIC.setGoal(desiredPos3, im_path_3)
            AIC.end_effector_selection(3)

        if count == 1600:
            AIC.setGoal(desiredPos1, im_path_2)
            AIC.end_effector_selection(2)

        if count == 2000:
            AIC.setGoal(desiredPos2, im_path_1)
            AIC.end_effector_selection(1)

        if count == 2400:
            AIC.setGoal(desiredPos, im_path_0)
            AIC.end_effector_selection(2)
        count += 1
        if count > 2999:
            AIC.plot_simulation_perception()
            sys.exit()

    rate.sleep()


if __name__ == "__main__":

   main()  
