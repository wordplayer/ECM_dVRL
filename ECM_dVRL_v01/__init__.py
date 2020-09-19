#!/usr/bin/env python
# coding: utf-8

from gym.envs.registration import registry, register, make, spec 
from ECM_dVRL_v01.ECM_dVRL import ECM_dVRL

register(id = 'ECM-v0',
        entry_point = 'ECM_dVRL_v01:ECM_dVRL',
        max_episode_steps = 200)



