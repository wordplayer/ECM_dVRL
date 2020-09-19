# ECM_dVRL
Automating the Endoscopic Camera Module on the da Vinci surgical system

Installations necessary before using this repository: 

CoppeliaSim from the Coppelia Robotics website (Access the educational version for free) <br/>
OpenAI Gym: `pip install gym` <br/>
PyRep: Clone the repository at `https://github.com/stepjam/PyRep.git` and follow the installation instructions on the PyRep GitHub page at `https://github.com/stepjam/PyRep` <br/>
OpenCV in Python: `pip install opencv-python` <br/>

Please note that the code in this repository has been developed and runs on Python 3.7<br/>

The source code for the da Vinci ECM - stationary PSM environment can be found in the `ECM_dVRL_v01` folder. <br/>

The environment can be created for training by following the steps: <br/>
`import gym` <br/>
`import ECM_dVRL_v01` <br/>
`env = gym.make('ECM-v0')` <br/>
