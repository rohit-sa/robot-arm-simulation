#Robot Arm Simulation

arm_arduino:
Contains code to be flashed into arduino Uno to enable transmission from simulation to actual robot arm.
Speific to motors and pin connections setup in the arm.

arm_model.py:
Main code handling simulation setup, running, creating dataset, testing results, vision based tracking, 
usb transmission to arduino and visualization.

nn_model.py:
Training of neural network for above model.

arm_env.yml:
Conda specific libraries used.