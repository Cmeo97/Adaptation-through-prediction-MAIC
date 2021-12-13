#!/usr/bin/env python3.7
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Float64, Float32MultiArray
from sensor_msgs.msg import JointState, Image
from gazebo_msgs.msg import LinkStates
import numpy as np
from numpy import linalg as LA
from AutoEncoder_sum import *
import matplotlib.image as img
from torch.autograd import Variable
import GPUtil
import matplotlib.pyplot as plt
import matplotlib
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.io
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, Matern
from joblib import dump, load
import ropy as rp


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AIC_agent:

    # Number of joints used
    N_JOINTS = 7

    # Image parameters
    IMG_WIDTH = 128
    IMG_HEIGHT = 128

    def __init__(self):

        # Initialise AE
        self.visual_AE = AutoEncoder()
        self.visual_AE.load_from_file("AutoEncoder")
        self.j_count = 0
        self.panda = rp.Panda()
        #Saturation Torque
        self.torque_max = np.zeros(7, float)
        self.torque_max = 50
 
        # Initialise Precision matrices 
        self.SigmaP_yq0 = np.zeros((self.N_JOINTS, self.N_JOINTS))
        self.SigmaP_v0 = np.zeros((self.N_JOINTS, self.N_JOINTS))
        self.SigmaP_yq1 = np.zeros((self.N_JOINTS, self.N_JOINTS))
        self.SigmaP_mu = np.zeros((self.N_JOINTS, self.N_JOINTS))
        self.SigmaP_muprime = np.zeros((self.N_JOINTS, self.N_JOINTS))
        self.SigmaP_v = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH))

        # Initialise beliefs 
        self.mu = np.zeros(self.N_JOINTS, float)
        self.mu_p = np.zeros(self.N_JOINTS, float)
        self.mu_pp = np.zeros(self.N_JOINTS, float)
        self.mu_dot = np.zeros(self.N_JOINTS, float)
        self.mu_dot_p = np.zeros(self.N_JOINTS, float)
        self.mu_dot_pp = np.zeros(self.N_JOINTS, float)
        self.EE = np.zeros((3, 1), float)

        # Initialise positions and velocities vectors 
        self.jointPos = np.zeros(self.N_JOINTS, float)
        self.jointPos_ = np.zeros(self.N_JOINTS, float)
        self.jointVel = np.zeros(self.N_JOINTS, float)
        #Initialise noises

        self.counter_noise = 0
        # Initialise  attractors
        self.mu_d = np.zeros(self.N_JOINTS, float)
        self.attractor_im = np.zeros((1, 1, self.IMG_HEIGHT, self.IMG_WIDTH))
        self.Im_attr = torch.tensor(np.float32(np.zeros((1, 1, 128, 128))),
                                    device=device, dtype=torch.float, requires_grad=True)

        # Initialise action
        self.act = np.zeros(self.N_JOINTS, float)
        self.act_ = np.zeros(self.N_JOINTS, float)
        self.Im_attr_numpy = np.zeros((128, 128), float)
        # Initialise network imput - output
        self.s_v_ = np.zeros((1, 1, self.IMG_HEIGHT, self.IMG_WIDTH))
        self.s_v = torch.tensor(np.float32(np.zeros((1, 1, 128, 128))),
                                device=device, dtype=torch.float, requires_grad=True)
        self.Im_hat = torch.tensor(np.zeros((1, 1, self.IMG_HEIGHT, self.IMG_WIDTH), float),
                                   device=device, dtype=torch.float, requires_grad=True)

        self.counter_image = 0
        #Gaussian Process model loading
        self.EE_data = scipy.io.loadmat('EE_1000.mat')
        self.Joints_data = scipy.io.loadmat('Joints_1000.mat')
        self.Joints_dataset = self.Joints_data['Joints']
        self.EE_dataset = self.EE_data['End_Effectors']
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], (1e-2, 0.4)) + WhiteKernel(0.001, [0.00001, 0.01])
        self.model = load('GPfit9261.joblib')
        self.params = self.model.kernel_.get_params()


        #Initialise flags and counters
        self.AIC_prop = True
        self.end_effector_counter = 0
        self.stochastic_counter = 0

        # Initialise publishers arguments
        self.AIC_mu = Float64MultiArray()
        dim = MultiArrayDimension()
        dim.label = "length"
        dim.size = 7
        self.AIC_mu.layout.dim.append(dim)
        
        self.AIC_mu_p = Float64MultiArray()
        self.AIC_mu_p.layout.dim.append(dim)
        self.AIC_mu_pp = Float64MultiArray()
        self.AIC_mu_pp.layout.dim.append(dim)
        
        self.F_Prop = Float64()

        self.tau1 = Float64()
        self.tau2 = Float64()
        self.tau3 = Float64()
        self.tau4 = Float64()
        self.tau5 = Float64()
        self.tau6 = Float64()
        self.tau7 = Float64()

        self.SPE = Float64MultiArray()
        dim_SPE = MultiArrayDimension()
        dim_SPE.label = "length"
        dim_SPE.size = 2
        self.SPE.layout.dim.append(dim_SPE)

        self.tau = Float64MultiArray()
        dim_tau = MultiArrayDimension()
        dim_tau.label = "length"
        dim_tau.size = 7
        self.tau.layout.dim.append(dim_tau)

        self.tauPub = rospy.Publisher("/RobotActions", Float64MultiArray, queue_size=20)

        #Subscribers
        sensorSub = rospy.Subscriber("/joint_states", JointState, self.jointStatesCallback_normal)
        sensorSub_noisy = rospy.Subscriber("/joint_states", JointState, self.jointStatesCallback)
        ImageSub = rospy.Subscriber('perception/Image', Image, self.image_callback)
        LinkSub = rospy.Subscriber("/robot1/panda_joint1_controller/command", Float64, self.linkStatesCallback)
        tau1Sub = rospy.Subscriber("/robot1/panda_joint2_controller/command", Float64, self.tau1Callback)
        tau2Sub = rospy.Subscriber("/robot1/panda_joint3_controller/command", Float64, self.tau2Callback)
        tau3Sub = rospy.Subscriber("/robot1/panda_joint3_controller/command", Float64, self.tau3Callback)
        tau4Sub = rospy.Subscriber("/robot1/panda_joint4_controller/command", Float64, self.tau4Callback)
        tau5Sub = rospy.Subscriber("/robot1/panda_joint5_controller/command", Float64, self.tau5Callback)
        tau6Sub = rospy.Subscriber("/robot1/panda_joint6_controller/command", Float64, self.tau6Callback)
        tau7Sub = rospy.Subscriber("/robot1/panda_joint7_controller/command", Float64, self.tau7Callback)
        GoalSub = rospy.Subscriber("/GoalPositions", Float64MultiArray, self.goal_callback)
     

        # Initialise STD_Im matrix
        self.Im_STD = np.zeros((1, 1, 128, 128))   # THAT'S STD MATRIX OF IMAGES
        self.Im_STD[0, 0] = np.loadtxt('Im_STD.csv', delimiter=',')
        self.Im_STD = torch.tensor(np.float32(self.Im_STD),
                                   device=device, dtype=torch.float, requires_grad=True)

        #MU_STD
        self.mu_STD = np.zeros(7)
        self.mu_STD = np.loadtxt('q_STD.csv', delimiter=',')
        
        self.mu_STD = torch.tensor(self.mu_STD, device=device, dtype=torch.float, requires_grad=True)

        # Initialise parameters
        # Attractor parameters 
        self.attractor_active = True
        self.beta = 1
        self.dataReceived = 0  
        self.dt_p = 0.001
        self.var_v_mu = 10
        self.sigma_v_a = 1*1e3
        self.sp_noise_variance = 0
        self.sigma_p = 1
        self.sigma_mu = 5

        #generative models gains
        self.k_f = 1
        self.k_g = 1

        # timestep
        self.dt = 0.001
        # Corrado's parameters
        self.var_mu = 5.0
        self.var_v = 15
        self.var_muprime = 5.0 
        self.var_q = 3 
        self.var_qdot = 3 
        self.k_mu = 18.67 
        self.k_a = 9
        # latent space parameters
        self.var_z1 = 1.0

        # Learning rates for the gradient descent
        self.k_a_ = 900
        self.err_flag = False
        
        for i in range(self.N_JOINTS):
            self.SigmaP_yq0[i, i] = 1/self.var_q
            self.SigmaP_yq1[i, i] = 1/self.var_qdot
            self.SigmaP_mu[i, i] = 1/self.var_mu
            self.SigmaP_muprime[i, i] = 1/self.var_muprime
            self.SigmaP_v0[i, i] = 1/self.var_v

        print('Initialized')
        GPUtil.showUtilization()



    #Callback functions

    def jointStatesCallback(self, msg):

        for i in range(self.N_JOINTS):
            self.jointPos[i] = msg.position[i]
            self.jointVel[i] = msg.velocity[i]


    def jointStatesCallback_normal(self, msg):

        for i in range(self.N_JOINTS):
            self.jointPos_[i] = msg.position[i]
            self.jointVel[i] = msg.velocity[i]


    def linkStatesCallback(self, msg):
        pose = msg.pose
        self.end_effector_x = pose[9].position.x
        self.end_effector_y = pose[9].position.y
        self.end_effector_z = pose[9].position.z
        self.EE[0, 0], self.EE[1, 0], self.EE[2, 0] = self.end_effector_x, self.end_effector_y, self.end_effector_z

        self.end_effector = np.sqrt(self.end_effector_x**2 + self.end_effector_y**2 + self.end_effector_z**2)

    def tau1Callback(self, msg):
        self.act_[0]=msg.data

    def tau2Callback(self, msg):
        self.act_[1]=msg.data

    def tau3Callback(self, msg):
        self.act_[2]=msg.data

    def tau4Callback(self, msg):
        self.act_[3]=msg.data

    def tau5Callback(self, msg):
        self.act_[4]=msg.data

    def tau6Callback(self, msg):
        self.act_[5]=msg.data

    def tau7Callback(self, msg):
        self.act_[6]=msg.data


    def image_callback(self, msg):
        Image_buffer = np.frombuffer(msg.data, dtype=np.uint8).reshape(128, 128)

        self.s_v_[0, 0] = Image_buffer / 255
        self.s_v = torch.tensor(self.s_v_, device=device, dtype=torch.float, requires_grad=True)

    def goal_callback(self, msg):

        self.setGoal(msg.data, self.counter_image)
        self.counter_image +=1 



    # Customized functions

    def end_effector_selection(self, p):
        self.end_effector_counter = p

    # Compute Gradients
    def get_latent_action(self):
        self.counter_noise = 0
        self.AIC_prop = False
        # Initialization
        self.Joints()
        # latent space
        self.z1 = torch.tensor(np.zeros((1, 16, 16), float), device=device, requires_grad=True)
        # Initialization current pose frame
        self.Im_hat = Variable(self.s_v, requires_grad=True)
        # get initial latent space
        self.Im_hat, self.mu_hat, self.z1 = self.visual_AE.prediction(self.joints,  self.Im_hat)
        # save initial latent space for latent interpolation
        self.act = self.act_         

    

    def get_df_dz1_attr(self):
        self.Im_hat, self.mu_hat, grad_Im, grad_mu = self.visual_AE.perception(self.z1, self.Im_STD, self.Im_attr, self.mu_STD, self.mu_attr)
        #Rescale beliefs
        self.mu[0] = self.mu_hat[0, 0].item() - 2.5
        self.mu[1] = self.mu_hat[0, 1].item() - 2.5
        self.mu[2] = self.mu_hat[0, 2].item() - 1.3
        self.mu[3] = self.mu_hat[0, 3].item() - 4.2
        self.mu[4] = self.mu_hat[0, 4].item() - 1.3
        self.mu[5] = self.mu_hat[0, 5].item() - 1.2
        self.mu[6] = self.mu_hat[0, 6].item() - 2.5
        k_im= 0.02
        k_mu= 1.5
        return k_im*grad_Im + k_mu*grad_mu


    def get_dg_dz1(self):
        self.Im_hat, self.mu_hat, grad_Im, grad_mu = self.visual_AE.perception(self.z1, self.Im_STD, self.s_v, self.mu_STD, self.joints)
        k_im = 0.02
        k_mu = 1.5
        return k_im*grad_Im + k_mu*grad_mu


    # Definitions
    def Joints(self):
        self.err_joints_real.append(self.jointPos - self.mu_d)
        self.err_images_real.append(LA.norm(self.s_v_ - self.Im_attr_numpy))
        self.joints = torch.tensor(self.jointPos, device=device, dtype=torch.float)
        self.joints[0] += 2.5
        self.joints[1] += 2.5
        self.joints[2] += 1.3
        self.joints[3] += 4.2
        self.joints[4] += 1.3
        self.joints[5] += 1.2
        self.joints[6] += 2.5

    def noise(self):
        self.jointPos[0] += self.noise_q0[self.counter_noise] 
        self.jointPos[1] += self.noise_q0[self.counter_noise] 
        self.jointPos[2] += self.noise_q0[self.counter_noise] 
        self.jointPos[3] += self.noise_q0[self.counter_noise] 
        self.jointPos[4] += self.noise_q0[self.counter_noise] 
        self.jointPos[5] += self.noise_q0[self.counter_noise] 
        self.jointPos[6] += self.noise_q0[self.counter_noise] 


    # Gaussian Process Regression
    def g_GPs(self):

        result = self.panda.jacob0(self.jointPos)[:-3, :].transpose().dot((self.panda.fkine(self.jointPos_).A.dot(np.array([0, 0, 0, 1]))[:-1] - self.model.predict(self.jointPos_.reshape(1, -1))).transpose())/self.var_v
        return result[:, 0]


    # MAIC-GP Algorithm
    def minimiseF_MAIF_GP(self, i):
        
        if i == 1:
            self.mu = self.jointPos
            self.mu_p = self.jointVel
        
        # Free-energy minimization using gradient descent and beliefs update
        self.mu_dot = self.mu_p - self.k_mu * (-self.SigmaP_yq0.dot(self.jointPos - self.mu) +
                                               self.SigmaP_mu.dot(
                                                   self.mu_p + self.mu - self.mu_d) + self.g_GPs())

        self.mu_dot_p = self.mu_pp - self.k_mu * (-self.SigmaP_yq1.dot((self.jointVel - self.mu_p).transpose()) +
                                                  self.SigmaP_mu.dot((self.mu_p + self.mu - self.mu_d).transpose()) +
                                                  self.SigmaP_muprime.dot((self.mu_pp + self.mu_p).transpose()))
        self.mu_dot_pp = - self.k_mu * (self.SigmaP_muprime.dot((self.mu_pp + self.mu_p).transpose()))

        # Belifs update
        self.mu = self.mu + self.dt_p * self.mu_dot  # Belief about the position
        self.mu_p = self.mu_p + self.dt_p * self.mu_dot_p  # Belief about motion of mu'
        self.mu_pp = self.mu_pp + self.dt_p * self.mu_dot_pp  # Belief about motion of mu''

        # Action update
        self.computeActions_PAIF()


    def minimiseF(self):
       
        self.Joints()
        self.z1_dot = self.k_g*self.get_dg_dz1() + self.k_f*self.get_df_dz1_attr()
        self.mu_dot = self.mu_p - self.k_mu*(-self.SigmaP_yq0.dot(self.jointPos-self.mu) +
                                             self.SigmaP_mu.dot(self.mu_p+self.mu-self.mu_d))
        self.mu_dot_p = self.mu_pp - self.k_mu*(-self.SigmaP_yq1.dot(self.jointVel-self.mu_p) +
                                                self.SigmaP_mu.dot(self.mu_p+self.mu-self.mu_d) +
                                                self.SigmaP_muprime.dot(self.mu_pp+self.mu_p))
        self.mu_dot_pp = - self.k_mu*(self.SigmaP_muprime.dot(self.mu_pp+self.mu_p))

        #Belifs update
        self.mu_p = self.mu_p + self.dt*self.mu_dot_p       # Belief about motion of mu'
        self.mu_pp = self.mu_pp + self.dt*self.mu_dot_pp    # Belief about motion of mu''
        self.z1 = Variable(self.z1 + self.z1_dot)
        #Action
        self.computeActions()
        self.err_perception[self.sim_counter] = Variable(self.joints - self.mu_hat) 
        self.err_attr[self.sim_counter] = self.mu_hat - self.mu_attr
        self.sim_counter += 1


    def computeActions(self):

        diff = (self.joints - self.visual_AE.Prop_Dec_action(self.z1)).detach().cpu().numpy()
        self.act = self.act - self.dt*self.k_a_*(self.SigmaP_yq1.dot(self.jointVel-self.mu_p) + self.SigmaP_yq0.dot(diff))

        
        for i in range(self.N_JOINTS):
            if self.act[i] < -self.torque_max[i]:
                    self.act[i] = -self.torque_max[i]
            else:
                if self.act[i] > self.torque_max[i]:
                        self.act[i] = self.torque_max[i]


        self.tau.data = self.act
        self.tauPub.publish(self.tau)
      
     
    def setGoal(self, desiredPos, im_path_counter):
        im_path = 'goal_poses/camera_image_' + str(self.paths[im_path_counter]) + '.jpeg'
        self.attractors = True
        for i in range(self.N_JOINTS):
            self.mu_d[i] = desiredPos[i]
        if im_path is not None:
            self.Im_attr[0] = torch.tensor(np.float32(img.imread(im_path)),
                                           device=device, dtype=torch.float, requires_grad=True)/255

        self.Im_attr_numpy = self.Im_attr[0, 0].detach().cpu().numpy()
       
        self.mu_attr = torch.tensor(self.mu_d, device=device, dtype=torch.float, requires_grad=True)
       
        self.mu_attr[0] += 2.5
        self.mu_attr[1] += 2.5
        self.mu_attr[2] += 1.3
        self.mu_attr[3] += 4.2
        self.mu_attr[4] += 1.3
        self.mu_attr[5] += 1.2
        self.mu_attr[6] += 2.5



    # Proprioceptive Active Inference et al. (Pezzato, 2019)
    def minimiseF_PAIF(self, i):

        if i == 0:
            self.mu = self.jointPos
            self.mu_p = self.jointVel

        # Compute single sensory prediction errors
        self.SPEq = (np.transpose(self.jointPos) - np.transpose(self.mu)).dot(self.SigmaP_yq0).dot(self.jointPos[:, None]-self.mu[:, None])
        self.SPEdq = (np.transpose(self.jointVel) - np.transpose(self.mu_p)).dot(self.SigmaP_yq1).dot(self.jointVel[:, None]-self.mu_p[:, None])
        self.SPEmu_p = (np.transpose(self.mu_p) + np.transpose(self.mu) - np.transpose(self.mu_d)).dot(self.SigmaP_mu).dot(self.mu_p[:, None]+self.mu[:, None]-self.mu_d[:, None])
        self.SPEmu_pp = (np.transpose(self.mu_pp) + np.transpose(self.mu_p)).dot(self.SigmaP_muprime).dot(self.mu_pp[:, None] + self.mu_p[:, None])

        # Free-energy as a sum of squared values (i.e. sum the SPE)
        self.F_Prop.data = self.SPEq + self.SPEdq + self.SPEmu_p + self.SPEmu_pp
        self.Fe_p.append(self.F_Prop.data)

        #Free-energy minimization using gradient descent and beliefs update
        self.mu_dot = self.mu_p - self.k_mu*(-self.SigmaP_yq0.dot(self.jointPos-self.mu) +
                                             self.SigmaP_mu.dot(self.mu_p+self.mu-self.mu_d))
        self.mu_dot_p = self.mu_pp - self.k_mu*(-self.SigmaP_yq1.dot(self.jointVel-self.mu_p) +
                                                self.SigmaP_mu.dot(self.mu_p+self.mu-self.mu_d) +
                                                self.SigmaP_muprime.dot(self.mu_pp+self.mu_p))
        self.mu_dot_pp = - self.k_mu*(self.SigmaP_muprime.dot(self.mu_pp+self.mu_p))

        #Belifs update
        self.mu = self.mu + self.dt_p * self.mu_dot            # Belief about the position
        self.mu_p = self.mu_p + self.dt_p*self.mu_dot_p       # Belief about motion of mu'
        self.mu_pp = self.mu_pp + self.dt_p*self.mu_dot_pp    # Belief about motion of mu''

        self.AIC_mu.data = self.mu
        self.AIC_mu_p.data = self.mu_p
        self.AIC_mu_pp.data = self.mu_pp

        matrix = np.zeros(2, float)
        matrix[0] = self.SPEq
        matrix[1] = self.SPEdq
        self.SPE.data = matrix
        #Action update
        self.computeActions_PAIF()
        self.IFE_pub.publish(self.F_Prop)
        # Sensory prediction error publisher
        self.SPE_pub.publish(self.SPE)

        # Publish beliefs
        self.beliefs_mu_pub.publish(self.AIC_mu)
        self.beliefs_mu_p_pub.publish(self.AIC_mu_p)
        self.beliefs_mu_pp_pub.publish(self.AIC_mu_pp)



    def computeActions_PAIF(self):

        self.act = self.act-self.dt_p*self.k_a_*(self.SigmaP_yq1.dot((self.jointVel-self.mu_p))+self.SigmaP_yq0.dot((self.jointPos-self.mu)))
      
        for i in range(self.N_JOINTS):
            if self.act[i] < -self.torque_max[i]:
                    self.act[i] = -self.torque_max[i]
            else:
                if self.act[i] > self.torque_max[i]:
                        self.act[i] = self.torque_max[i]
 
        self.tau.data = self.act
        self.tauPub.publish(self.tau)


    #Imaginary simulation
    def get_latent_perception(self, desiredPos2, im_path_2):
        # Initialization
        self.joints = torch.tensor(desiredPos2, device=device, dtype=torch.float, requires_grad=True)
        self.joints[0] += 2.5
        self.joints[1] += 2.5
        self.joints[2] += 1.3
        self.joints[3] += 4.2
        self.joints[4] += 1.3
        self.joints[5] += 1.2
        self.joints[6] += 2.5
        # latent space
        self.z1 = torch.tensor(np.zeros((1, 16, 16), float), device=device, requires_grad=True)

        
        self.Im_hat[0, 0] = torch.tensor(np.float32(img.imread(im_path_2)), device=device, dtype=torch.float,
                                         requires_grad=True) / 255

        self.Im_hat, self.mu_hat, self.z1 = self.visual_AE.prediction(self.joints, self.Im_hat)
        print(self.mu_hat)

        # save initial latent space for latent interpolation
        self.z1_init = self.z1


    def logger(self):

      Err_joints = np.array(self.err_joints_real)
      Err_image = np.array(self.err_images_real)
      Err_perception = self.err_perception.detach().cpu().numpy()
      Err_attr = self.err_attr.detach().cpu().numpy()
 
      np.savetxt('err_joints.csv', np.mean(Err_joints, 1), delimiter=',')
      np.savetxt('err_image.csv', Err_image, delimiter=',')
      np.savetxt('err_perception.csv', np.mean(Err_perception, 1), delimiter=',')
      np.savetxt('err_attr.csv', np.mean(Err_attr, 1), delimiter=',')


    def perception(self, i):
      self.Im_hat, self.mu_hat, grad_Im, grad_mu = self.visual_AE.perception(self.z1, self.Im_STD, self.Im_attr, self.mu_STD, self.mu_attr)
      self.z1_dot = (0.02*grad_Im + 1.5*grad_mu)
      self.z1 = self.z1 + self.z1_dot
      self.err_joints[i, :] = (self.mu_hat - self.mu_attr).detach().cpu().numpy()
      self.err_Im[i] = (self.Im_hat - self.Im_attr).detach().cpu().numpy()



