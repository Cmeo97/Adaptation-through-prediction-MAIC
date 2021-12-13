# Adaptation through prediction: multisensory active inference torque control

Submitted to IEEE Transactions on Cognitive and Developmental Systems

Abstract:
Adaptation to external and internal changes is major for robotic systems in uncertain environments. Here we present a novel multisensory active inference torque controller for industrial arms that shows how prediction can be used to resolve adaptation. Our controller, inspired by the predictive brain hypothesis, improves the capabilities of current active inference approaches by incorporating learning and multimodal integration of low and high-dimensional sensor inputs (e.g., raw images) while simplifying the architecture. We performed a systematic evaluation of our model on a 7DoF Franka Emika Panda robot arm by comparing its behavior with previous active inference baselines and classic controllers, analyzing both qualitatively and quantitatively adaptation capabilities and control accuracy. Results showed improved control accuracy in goal-directed reaching with high noise rejection due to multimodal filtering, and adaptability to dynamical inertial changes, elasticity constraints and human disturbances without the need to relearn the model nor parameter retuning.


# User guide:

- Hardware Required:
     - Franka Emika Panda robot arm
     - Camera 

- Packages Required:
     - Franka ROS: https://github.com/frankaemika/franka_ros
     - Camera driver 
 
- Installation: 
     - a unique catkin_ws must be created containing MAIC_src, Camera_src and franka interface controller_src folders. 
  

