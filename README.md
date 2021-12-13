# Adaptation through prediction: multisensory active inference torque control

Submitted to IEEE Transactions on Cognitive and Developmental Systems

Abstract:
Adaptation to external and internal changes is major for robotic systems in uncertain environments. Here we present a novel multisensory active inference torque controller for industrial arms that shows how prediction can be used to resolve adaptation. Our controller, inspired by the predictive brain hypothesis, improves the capabilities of current active inference approaches by incorporating learning and multimodal integration of low and high-dimensional sensor inputs (e.g., raw images) while simplifying the architecture. We performed a systematic evaluation of our model on a 7DoF Franka Emika Panda robot arm by comparing its behavior with previous active inference baselines and classic controllers, analyzing both qualitatively and quantitatively adaptation capabilities and control accuracy. Results showed improved control accuracy in goal-directed reaching with high noise rejection due to multimodal filtering, and adaptability to dynamical inertial changes, elasticity constraints and human disturbances without the need to relearn the model nor parameter retuning.


# User guide:


## Hardware Required:
- Franka Emika Panda robot arm
- Camera 

## Requirements
- ROS (melodic)
- pytorch 1.7.0
- cv2
- PyRobot
- sklearn
- seaborn 
- Franka ROS
- Camera driver 
## Installation
Once the dependencies are installed, a catkin workspace has to be created. To do it:

- Create a catkin_ws folder: $ mkdir -p catkin_ws/src
- Move to the folder: $ cd catkin_ws/src
- Clone the repository $ git clone https://github.com/Cmeo97/Adaptation-through-prediction-MAIC
- Clone franka interface repository
- Clone used camera driver repository 
- Move back to catkin_ws: $ cd ..
- Build the workspace: $ catkin_make
- Source: $ source devel/setup.bash

remember to change the subscribers topic names based on your publishers names. 

## Running the code
To run the controller:

- After building and sorcing the workspace you have to launch the franka interface which has to publish the joint states.
- run the camera launcher 

- Go to the controller folder: $ cd src/MAIC/src
- You have to run the camera node, which subscribe images and publish them for the controller: $ python2.7 camera_launcher.py
- then, in another terminal, run the controller: $ python MAIC-#.py  (# can be either GP or VAE)

- To run the Brain simulation run: $ python Brain_Simulation.py


