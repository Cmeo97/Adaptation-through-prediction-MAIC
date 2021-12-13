#!/usr/bin/env python2.7
import argparse
from MAIC_full import *
import gc
import GPUtil
import time
import subprocess
import scipy.io
from multiprocessing import Process

robot = 1
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

    AIC.setGoal(desiredPos, '')

  
    rate = rospy.Rate(120)
    count = 1
    while not rospy.is_shutdown():


        if count > 0:
            p2 = Process(target=AIC.minimiseF_MAIF_GP(count))

        count += 1
        if count == 4000:
            AIC.setGoal(desiredPos1, '')
            AIC.end_effector_selection(1)

        if count == 8000:
            AIC.setGoal(desiredPos2, '')
            AIC.end_effector_selection(2)

        if count == 12000:
            AIC.setGoal(desiredPos3, '')
            AIC.end_effector_selection(3)

        if count == 16000:
            AIC.setGoal(desiredPos2, '')
            AIC.end_effector_selection(4)

        if count == 20000:
            AIC.setGoal(desiredPos1, '')
            AIC.end_effector_selection(5)

        if count > 23999:
            AIC.logger()
            sys.exit()

    rate.sleep()


if __name__ == "__main__":
    main()
