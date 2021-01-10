#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyrep.pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from pyrep.misc.signals import IntegerSignal
import numpy as np
import transforms3d.quaternions as quaternions


# In[2]:


class ArmPSM(PyRep):
    def __init__(self, pr, armNumber = 1):
        """self.pr = PyRep()
        self.pr.launch(scenePath)
        self.pr.start()
        self.pr.step()"""
        self.pr = pr
        self.psm = armNumber
        self.ik_mode = 1
        
        self.base_handle = Shape('RCM_PSM{}'.format(self.psm))
        self.j1_handle   = Joint('J1_PSM{}'.format(self.psm))
        self.j2_handle   = Joint('J2_PSM{}'.format(self.psm))
        self.j3_handle   = Joint('J3_PSM{}'.format(self.psm))
        self.j4_handle   = Joint('J1_TOOL{}'.format(self.psm))
        self.j5_handle   = Joint('J2_TOOL{}'.format(self.psm))
        self.j6d_handle  = Joint('J3_dx_TOOL{}'.format(self.psm))
        self.j6s_handle  = Joint('J3_sx_TOOL{}'.format(self.psm))

        self.j5_dummy_handle   = Dummy('J2_virtual_TOOL{}'.format(self.psm))

        self.j6d_tip_dummy_handle   = Dummy('J3_dx_tip_TOOL{}'.format(self.psm))
        self.j6s_tip_dummy_handle   = Dummy('J3_sx_tip_TOOL{}'.format(self.psm))

        self.ik_target_dx_dummy_handle = Dummy('IK_target_dx_PSM{}'.format(self.psm))
        self.ik_target_sx_dummy_handle = Dummy('IK_target_sx_PSM{}'.format(self.psm))
        self.marker = Shape('yaw_{}'.format(self.psm))
        self.EE_virtual_handle = Dummy('EE_virtual_TOOL{}'.format(self.psm))
        
        self.ik_signal = IntegerSignal("run_IK_PSM{}".format(self.psm))
        self.dyn_signal = IntegerSignal("run_dyn_PSM{}".format(self.psm))
        
        #Set IK mode off to save on computation for VREP:
        self.setIkMode(0)
        #Set dynamics mode off to save on compuation time for VREP:
        self.setDynamicsMode(0)
        
        self.IK_PSM1_group = sim.simGetIkGroupHandle('IK_PSM1')

    #dyn_mode = 1 turns on dynamics
    #dyn_mode = 0 turns off dynamics
    def setDynamicsMode(self, dyn_mode):
        self.dyn_mode = dyn_mode
        self.dyn_signal.set(self.dyn_mode)
        
    #ik_mode = 1 turns on ik_mode
    #ik_mode = 0 turns off ik_mode
    def setIkMode(self, ik_mode):
        self.ik_mode = ik_mode
        self.ik_signal.set(self.ik_mode)
        
    def posquat2Matrix(self, pos, quat):
        T = np.eye(4)
        T[0:3, 0:3] = quaternions.quat2mat([quat[-1], quat[0], quat[1], quat[2]])
        T[0:3, 3] = pos

        return np.array(T)

    def matrix2posquat(self,T):
        pos = T[0:3, 3]
        quat = quaternions.mat2quat(T[0:3, 0:3])
        quat = [quat[1], quat[2], quat[3], quat[0]]

        return np.array(pos), np.array(quat)

    def getJawAngle(self):
        pos6d = self.j6d_handle.get_joint_position()
        pos6s = self.j6s_handle.get_joint_position()
        jawAngle = 0.5 * (pos6d + pos6s)/0.4106
        return jawAngle
    
    def getJointAngles(self):
        pos1  = self.j1_handle.get_joint_position()
        pos2  = self.j2_handle.get_joint_position()
        pos3  = self.j3_handle.get_joint_position()
        pos4  = self.j4_handle.get_joint_position()
        pos5  = self.j5_handle.get_joint_position()
        pos6s = self.j6s_handle.get_joint_position()
        pos6d = self.j6d_handle.get_joint_position()

        pos6     = 0.5*(pos6d - pos6s)
        jawAngle = 0.5*(pos6d + pos6s)/0.4106

        jointAngles = np.array([pos1, pos2, pos3, pos4, pos5, pos6])

        return jointAngles, jawAngle
    
    def getJointVelocities(self):
        vel1  = self.j1_handle.get_joint_velocity()
        vel2  = self.j2_handle.get_joint_velocity()
        vel3  = self.j3_handle.get_joint_velocity()
        vel4  = self.j4_handle.get_joint_velocity()
        vel5  = self.j5_handle.get_joint_velocity()
        vel6s = self.j6s_handle.get_joint_velocity()
        vel6d = self.j6d_handle.get_joint_velocity()

        vel6   = 0.5*(vel6s - vel6d)
        jawVel = 0.5*(vel6s + vel6d)/0.4106

        jointVelocities = np.array([vel1, vel2, vel3, vel4, vel5, vel6])

        return jointVelocities, jawVel
    
    def setJointAngles(self, jointAngles, jawAngle):

        self.j1_handle.set_joint_position(jointAngles[0])
        self.j2_handle.set_joint_position(jointAngles[1])
        self.j3_handle.set_joint_position(jointAngles[2])
        self.j4_handle.set_joint_position(jointAngles[3])
        self.j5_handle.set_joint_position(jointAngles[4])

        pos6s = 0.4106*jawAngle - jointAngles[5]
        pos6d = 0.4106*jawAngle + jointAngles[5]

        self.j6s_handle.set_joint_position(pos6s)
        self.j6d_handle.set_joint_position(pos6d)
    
    def getPoseAtJoint(self, j):
        if j == 0:
            pose = self.base_handle.get_pose()
            pos, quat = pose[0:3], pose[3:]
        elif j == 1:
            pose = self.j2_handle.get_pose(relative_to = self.base_handle)
            pos, quat = pose[0:3], pose[3:]
            T = self.posquat2Matrix(pos,quat)
            rot90x = [[1, 0,  0, 0], 
                      [0, 0, -1, 0], 
					  [0, 1,  0, 0],
					  [0, 0,  0, 1]]
            pos, quat = self.matrix2posquat(np.dot(T, rot90x))
        elif j == 2:
            pose = self.j3_handle.get_pose(relative_to = self.base_handle)
            pos, quat = pose[0:3], pose[3:]
            T = self.posquat2Matrix(pos,quat)
            rot    = [[0,  0,  1, 0], 
					  [-1, 0,  0, 0], 
					  [0, -1,  0, 0],
					  [0,  0,  0, 1]]
            pos, quat = self.matrix2posquat(np.dot(T, rot))
        elif j == 3:
            pose = self.j4_handle.get_pose(relative_to = self.base_handle)
            pos, quat = pose[0:3], pose[3:]
            T = self.posquat2Matrix(pos,quat)
            rot    = [[-1, 0,  0, 0,], 
					  [0, -1,  0, 0], 
					  [0,  0,  1, 0],
					  [0,  0,  0, 1]]
            pos, quat = self.matrix2posquat(np.dot(T, rot))
        elif j == 4:
            pose = self.j5_handle.get_pose(relative_to = self.base_handle)
            pos, quat = pose[0:3], pose[3:]
            T = self.posquat2Matrix(pos,quat)
            rot    = [[0, 0, -1, 0], 
					  [1, 0,  0, 0], 
					  [0,-1,  0, 0],
					  [0, 0,  0, 1]]
            pos, quat = self.matrix2posquat(np.dot(T, rot))
        elif j == 5:
            pose = self.j5_dummy_handle.get_pose(relative_to = self.base_handle)
            pos, quat = pose[0:3], pose[3:]
        else:
            pose = self.EE_virtual_handle.get_pose(relative_to = self.base_handle)
            pos, quat = pose[0:3], pose[3:]
            if j != 6:
                T = self.posquat2Matrix(pos,quat)

                ct = np.cos(0)
                st = np.sin(0)

                ca = np.cos(-np.pi/2.0)
                sa = np.sin(-np.pi/2.0)

                T_x = np.array([[1,  0,  0, 0],
				               [0, ca, -sa, 0 ],
				               [0, sa,  ca, 0 ],
				               [0, 0, 0,    1 ]])
                T_z = np.array([[ct, -st, 0, 0],
				                [st,  ct, 0, 0],
				                [0,    0, 1, 0.0102],
				                [0,    0, 0, 1]])
                T = np.dot(np.dot(T,T_x), T_z)

                pos, quat = self.matrix2posquat(T)

        return np.array(pos), np.array(quat)
    
    def getPoseAtEE(self):
        return self.getPoseAtJoint(6)
    
    #def getVelocityAtEE(self):
     #   return self.EE_virtual_handle.get_velocity()
    
    def setPoseAtEE(self, pos, quat, jawAngle):
        theta = 0.4106*jawAngle

        b_T_ee = self.posquat2Matrix(pos, quat)

        ee_T_sx = np.array([[ 9.99191168e-01,  4.02120491e-02, -5.31786338e-06,4.17232513e-07],
				       [-4.01793160e-02,  9.98383134e-01,  4.02087139e-02, -1.16467476e-04],
				       [ 1.62218404e-03, -4.01759782e-02,  9.99191303e-01, -3.61323357e-04],
				       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,1.00000000e+00]])

        ee_T_dx = np.array([[-9.99191251e-01, -4.02099858e-02, -1.98098369e-06, 4.17232513e-07],
				       [-4.01773877e-02,  9.98383193e-01, -4.02091818e-02, -1.16467476e-04],
				       [ 1.61878841e-03, -4.01765831e-02, -9.99191284e-01, -3.61323357e-04],
				       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

        b_T_sx = np.dot(b_T_ee, ee_T_sx)
        b_T_dx = np.dot(b_T_ee, ee_T_dx)

        ct = np.cos(theta)
        st = np.sin(theta)

        x_T_ts = np.array([[ct, -st, 0,  -st*0.009,],
						 [st,    ct, 0,   ct*0.009],
						 [0 ,   0, 1, 0,],
						 [0,    0, 0, 1]])

        pos_sx, quat_sx = self.matrix2posquat(np.dot(b_T_sx, x_T_ts))
        pos_dx, quat_dx = self.matrix2posquat(np.dot(b_T_dx, x_T_ts))

        self.ik_target_dx_dummy_handle.set_pose(np.r_[pos_dx, quat_dx], relative_to = self.base_handle)
        self.ik_target_sx_dummy_handle.set_pose(np.r_[pos_sx, quat_sx], relative_to = self.base_handle)

    def get_marker_position(self, relative_to = None):
        return self.marker.get_position(relative_to)

    def getJacobian(self):
        sim.simCheckIkGroup(self.IK_PSM1_group, [
            self.j1_handle.get_handle(), self.j2_handle.get_handle(), 
            self.j3_handle.get_handle(), self.j4_handle.get_handle(),
            self.j5_handle.get_handle(), self.j6d_handle.get_handle(), self.j6s_handle.get_handle()])
        J, (rows, cols) = sim.simGetIkGroupMatrix(self.IK_PSM1_group, 0)
        J = np.array(J).reshape((rows, cols), order = 'F')
        return J
    """def stopSim(self):
        self.pr.stop()
        self.pr.shutdown()"""




