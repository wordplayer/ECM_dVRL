#!/usr/bin/env python
# coding: utf-8

import gym
from ECM_dVRL_v01.ECMEnv_model import ECMEnv

class ECM_dVRL(ECMEnv):
    def __init__(self):
        super(ECM_dVRL, self).__init__(1, 4, 4, 2, 100, False, 
                                 '/home/arclab-flo/Desktop/Bose/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04/scenes/single_arm_one_marker (copy).ttt')




