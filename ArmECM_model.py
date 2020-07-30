#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyrep.pyrep import PyRep
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np


# In[2]:


class ArmECM(PyRep):
    def __init__(self, pr):
        
        #super(ArmECM, self).__init__(scenePath)
        """self.pr = PyRep()
        self.pr.launch(scenePath)
        self.pr.start()
        self.pr.step()"""
        self.pr = pr
        self.base_handle = Shape('L0_respondable_ECM')
        self.j1_handle = Joint('J1_ECM')
        self.j2_handle = Joint('J2_ECM')
        self.j3_handle = Joint('J3_ECM')
        self.j4_handle = Joint('J4_ECM')
        
        self.left_cam = VisionSensor('Vision_sensor_left')
        self.right_cam = VisionSensor('Vision_sensor_right')
        
    def getJointAngles(self):
        
        pos1 = self.j1_handle.get_joint_position()
        pos2 = self.j2_handle.get_joint_position()
        pos3 = self.j3_handle.get_joint_position()
        pos4 = self.j4_handle.get_joint_position()
        return np.array([pos1, pos2, pos3, pos4])
    
    def setJointAngles(self, jointAngles):
        
        self.j1_handle.set_joint_position(jointAngles[0])
        self.j2_handle.set_joint_position(jointAngles[1])
        self.j3_handle.set_joint_position(jointAngles[2])
        self.j4_handle.set_joint_position(jointAngles[3])
        
    def getCurrentPose(self):
        
        return self.left_cam.get_pose(relative_to = self.base_handle)
    
    def getStereoImagePairs(self):
        return np.fliplr(self.left_cam.capture_rgb()), np.fliplr(self.right_cam.capture_rgb())
    
    """def stopSim(self):
        self.pr.stop()
        self.pr.shutdown()"""


