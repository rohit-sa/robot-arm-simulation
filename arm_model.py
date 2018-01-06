# -*- coding: utf-8 -*-
"""
@author: Rohit
"""
from random import randint
from vpython import *
import numpy as np
import pandas as pd
import serial as sl
import time as tm
import struct as st
import cv2
import math
import matplotlib.pyplot as plt
#import keras.backend as K

"""
Custom Classes
"""
class Transmission(object):
    """    
    function:    def __init__    
    input parameters: self,port='COM6'    
    Notes:     
    """    
    def __init__(self,port='COM6'):
        self.__outgoing_data = b''
        self.__connection = 0
        try:
            self.__connection = sl.Serial(port=port,baudrate=9600,timeout=0.1)
        except(sl.SerialException):
            print('Nothing on '+port)
        return
    
    """    
    function:    def __convert_angles    
    input parameters: self,data    
    Notes:     
    """    
    def __convert_angles(self,data):
        out_angle = []
        out_angle.append(data[0]+90)
        out_angle.append(data[1])
        out_angle.append(abs(data[2]-90))
        out_angle.append(-data[3])
        out_angle.append(data[4])
        return out_angle
    
    """    
    function:    def transmit    
    input parameters: self,data    
    Notes:     
    """    
    def transmit(self,data):
        if self.__connection == 0:
            return
        out_angle = self.__convert_angles(data) 
        out_data_packet = [st.pack('>B',int(angle)) for angle in out_angle]
        self.__connection.write(self.__outgoing_data.join(out_data_packet))
        tm.sleep(0.1)
        return
    
"""
class " GUI "
Notes: creates gui to control angles for limbs
"""     
class GUI(object):
    """
    function " __init__ "
    Notes: uses model to intialize angles and limits and states
    """ 
    def __init__(self,model):
        self.__angles = model['angles']
        self.__limits = model['limits']
        self.__s_id = []
        self.__b_id = []
        self.__claw = 0 #open
        self.__state = True #running
    """
    function " s_callback "
    Notes: callback function for sliders for angles
    """     
    def s_callback(self,s):
        i = s.id
        u = self.__limits['u'][i]
        l = self.__limits['l'][i]
        self.__angles[i] = self.__s_id[i].value*(u-l) + l
    """
    function " b_callback "
    Notes: button callback function for clicking
    """         
    def b_callback(self,b):
        if b.id == 0:
            if self.__state == True:
                self.__b_id[0].text ='<b>Run</b>' 
            else:
                self.__b_id[0].text ='<b>Pause</b>' 
            self.__state = not self.__state
        else:
            if self.__claw == 0:
                self.__b_id[1].text = '<b>Release</b>'
                self.__claw = 1
            else:
                self.__b_id[1].text = '<b>Grab</b>'
                self.__claw = 0
    """
    function " build "
    Notes: cerates the gui elements (sliders and button)
    """             
    def build(self):
        self.__b_id.append(button(text='<b>Pause</b>',bind=self.b_callback, id=0))
        scene.append_to_caption('\t')
        self.__b_id.append(button(text='<b>Grab</b>',bind=self.b_callback, id=1))
        scene.append_to_caption('\n\n')
        for i in range(len(self.__angles)):
            caption = '\tJoint ' + str(i+1)
            self.__s_id.append( slider(length=250, bind = self.s_callback, id=i))
            scene.append_to_caption(caption + '\n\n')
    """
    function " read_values "
    Notes: get state of model angles and claw state
    """    
    def read_values(self):
        return self.__angles, self.__claw
    """
    function " read_state "
    Notes: get running state
    """ 
    def read_state(self):
        return self.__state
    
class Arm_segment(object):
    """    
    function:    def __init__    
    input parameters: self, pos, length    
    Notes:     
    """    
    def __init__(self, pos, length):
        self.pos = pos
        self.l = length
    """
    Generate a single arm segement (ball+cylinder)
    Needed to be run during model building/initialization
    """
    def generate(self,angle,phi,pos=None):
        new_pos, axis = self.calculate_pos(angle,phi,pos)
        arm = cylinder(pos=self.pos,axis=axis,radius=0.5)
        joint = sphere(pos=self.pos,radius=0.7,color=color.cyan)
        limb = {'arm':arm, 'joint':joint}
        return limb, new_pos
    
    """
    Calulate end points of arm segments
    """
    def calculate_pos(self,angle,phi,pos=None):
        new_pos = vector(0,0,0)
        if pos != None:
            self.pos = pos
        angle = np.radians(angle)
        new_pos.x = self.pos.x + self.l*np.cos(angle)*np.cos(np.radians(phi))
        new_pos.z = self.pos.z + self.l*np.cos(angle)*np.sin(np.radians(phi))
        new_pos.y = self.pos.y + (self.l*np.sin(angle))
        axis = self.l * (new_pos - self.pos).norm() 
        return new_pos, axis

"""
Camera tracking
"""
class Obj_tracking(object):
    """    
    function:    def __init__    
    input parameters: self,model    
    Notes:     
    """    
    def __init__(self,model):
        self.__total_l = sum(model['lengths'])
        self.__cap = cv2.VideoCapture(0)
        print(self.__cap.get(cv2.CAP_PROP_FPS))
        self.__low_thresh = (50,100,100)
        self.__high_thresh = (70,255,255)
        self.__kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        self.__target = sphere(pos=vector(0,0,0),radius=0.75,color=color.red)
        self.__target.visible =False
        self.__marked = False
        self.__track_algo = 'KCF'
        self.__tracker = cv2.Tracker_create(self.__track_algo)
    
    """    
    function:    def __detect_target    
    input parameters: self,frame,draw    
    Notes:     
    """    
    def __detect_target(self,frame,draw):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame,self.__low_thresh,self.__high_thresh)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,self.__kernel)
        im,contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        (x,y,w,h) = (-1,0,0,0)
        if len(contours) != 0:
            contour_areas = [cv2.contourArea(contour) for contour in contours]
            max_index = np.argmax(contour_areas)
            x,y,w,h = cv2.boundingRect(contours[max_index])
            if draw:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        return (x,y,w,h)
    
    """    
    function:    def __generate_target    
    input parameters: self,target_pos    
    Notes:     
    """    
    def __generate_target(self,target_pos):
        pos = vector(0,0,0)
        pos.x = (1-target_pos[2])*self.__total_l
        pos.y = (1-target_pos[1])*self.__total_l
        pos.z = (target_pos[0]-0.5)*self.__total_l*2
        self.__target.pos=pos
        self.__target.visible = True
        return
    
    """    
    function:    def track    
    input parameters: self,found,draw=False    
    Notes:     
    """    
    def track(self,found,draw=False):
        ret,frame = self.__cap.read()
        if not self.__marked:
            self.__target_box = self.__detect_target(frame,draw)  
            if self.__target_box != (-1,0,0,0) and found:
                self.__marked = self.__tracker.init(frame, self.__target_box)
        else:
            _, self.__target_box = self.__tracker.update(frame)
            if draw:
                p1 = (int(self.__target_box[0]), int(self.__target_box[1]))
                p2 = (int(self.__target_box[0] + self.__target_box[2]), int(self.__target_box[1] + self.__target_box[3]))
                cv2.rectangle(frame, p1, p2, (0,0,255))
            height,width = frame.shape[0],frame.shape[1]
            x = self.__target_box[0]/width
            y = self.__target_box[1]/height
            z =  0.2
            self.__generate_target((x,y,z))
        
        cv2.imshow('Out',frame)
        return self.__target
    
    """    
    function:    def reset    
    input parameters: self    
    Notes:     
    """    
    def reset(self):
        self.__marked = False
        self.__tracker = cv2.Tracker_create(self.__track_algo)
    
    """    
    function:    def terminate    
    input parameters: self    
    Notes:     
    """    
    def terminate(self):
        self.__cap.release()
        cv2.destroyAllWindows()
        return   

"""
Dataset creation
"""
class Dataset(object):
    """    
    function:    def __init__    
    input parameters: self    
    Notes:     
    """    
    def __init__(self):
        self.__dataset = pd.DataFrame(columns=['x','y','z','angle0','angle1','angle2','angle3'])
        return
    
    """    
    function:    def update_dataset    
    input parameters: self,model,target_pos    
    Notes:     
    """    
    def update_dataset(self,model,target_pos):
        data_row = {'x':target_pos.x, 'y':target_pos.y,'z':target_pos.z}
        angles = model['angles']
        for i in range(len(angles)):
            data_row.update({'angle'+str(i):angles[i]})
        self.__dataset = self.__dataset.append(data_row, ignore_index=True)
        return
    
    """    
    function:    def save_dataset    
    input parameters: self,filename='dataset.csv'    
    Notes:     
    """    
    def save_dataset(self,filename='dataset.csv'):
        self.__dataset = self.__dataset.set_index(self.__dataset.columns[0])
        self.__dataset.to_csv(filename)
        return
    
    """    
    function:    def get_dataset    
    input parameters: self    
    Notes:     
    """    
    def get_dataset(self):
        return self.__dataset


"""
Setup vpython scene, camera position
create axes and colors
"""

"""
function: create_scene
input parameters: height=None
Notes: 
"""
def create_scene(height=None):
    if height==None:
        scene.height = 600
    else:
        scene.height = height
    scene.width = 800
    scene.forward = vector(-1.344230, -1.055753, -0.280052)
    size = 30
    x_axis = cylinder(pos=vector(0,0,0),axis=vector(size,0,0),radius=0.05, color=color.red)
    L_x_axis = label(pos=x_axis.axis,text='x',xoffset=1,yoffset=1, box=False)
    y_axis = cylinder(pos=vector(0,0,0),axis=vector(0,size,0),radius=0.05, color=color.red)
    L_y_axis = label(pos=y_axis.axis,text='y',xoffset=1,yoffset=1, box=False)
    z_axis = cylinder(pos=vector(0,0,0),axis=vector(0,0,size),radius=0.05, color=color.red)
    L_z_axis = label(pos=z_axis.axis,text='z',xoffset=1,yoffset=1, box=False)
    base = box(pos=vector(0,0,0), length=size, height=0.1, width=size) 
    base.color = color.green
    base.opacity = 0.4
    return
   
"""
Calulates angles with refernce to x-z axis
"""     
def update_angles(angles,limits):
    for i in range(len(angles)):
        if angles[i] < limits['l'][i] :
            angles[i] = limits['l'][i]
        elif angles[i] > limits['u'][i]:
            angles[i] = limits['u'][i]
        else:
            pass
    total_angles = [angles[0],angles[1]]
    for i in range(2,len(angles)):
        total_angles.append(angles[i] + total_angles[i-1])
    return total_angles

"""
Repositions arm segments
"""   
def update(model):
    seg, arm, angles, pos = model['seg'],model['arm'],model['angles'],model['pos']
    limits = model['limits']
    total_angles = update_angles(angles,limits)
    claw_state = model['claw_state']
    joint = model['joint']
    for i in range(len(arm)):
        arm[i].pos = pos[i]
        joint[i].pos = pos[i]
        new_pos, axis = seg[i].calculate_pos(total_angles[i+1],phi=total_angles[0],pos=pos[i])
        arm[i].axis = axis
        pos[i+1] = new_pos
    joint[-1].pos = new_pos
    if claw_state == 1:
        joint[-1].color = color.red
    else:
        joint[-1].color = color.blue
    return pos[-1]    

"""
Build/Initialize the arm
Creates a dictionary with arm properties
length, limits, current position of arm segments
current angles, claw/grabber state
"""
       
"""
function: init_model
input parameters: lengths=None, limits=None
Notes: 
"""
def init_model(lengths=None, limits=None):
    if lengths == None:
        lengths = [3,6,5,4]
    if limits == None:
        limits = {'l':[-90,0,-90,-90],'u':[90,90,90,0]}
    angles = [0,45,-15,-20]
    total_angles = update_angles(angles,limits)
    """
    Base joint creations
    """
    a = cylinder(pos=vector(0,0,0),axis=vector(0,1,0),length=lengths[0],radius=0.5)
    a1_joint = sphere(pos=vector(0,0,0),radius=0.7,color=color.cyan)  
    pos=[a.axis]
    seg = []
    arm = []
    joint = []
    j = 0
    for i in range(1,len(total_angles)):
        seg.append(Arm_segment(pos=pos[j],length=lengths[i]))
        limb, new_pos = seg[j].generate(total_angles[i],phi=total_angles[0])
        pos.append(new_pos)
        arm.append(limb['arm'])
        joint.append(limb['joint'])
        j+=1
    joint.append(sphere(pos=pos[-1],radius=0.7,color=color.blue))
    model = {'pos':pos, 'seg':seg, 'arm':arm,'joint':joint,'angles':angles}
    model.update({'lengths':lengths,'limits':limits, 'claw_state':0})
    return model

"""
Testing model movement
"""
def rand_move_model(model):
    limits = model['limits']
    angles = model['angles']
    for i in range(len(angles)):
        if angles[i] < limits['l'][i] :
            angles[i] += randint(0,5)
        elif angles[i] > limits['u'][i]:
            angles[i] -= randint(0,5)
        else:
            angles[i] += randint(-5,5)
    model['angles'] = angles
    return model
  
"""
Calculates center of mass of arm
"""    
def com_vis(model,com=None):
    joint = model['joint']
    length = model['lengths']
    com_pos = vector(0,0,0)
    for i in range(len(joint)):
        com_pos += joint[i].pos 
    com_pos = com_pos/len(joint)
    if com == None:
        com = sphere(pos=com_pos,radius=0.8,color=color.magenta)
    else:
        com.pos = com_pos
    return com


"""
create specific path
Generates random target location using lengths of arm segment
"""
class TargetGenerator(object):
    """    
    function:    def __init__    
    input parameters: self,model,path='circle'    
    Notes:     
    """    
    def __init__(self,model,path='circle'):
        self.__total_l = sum(model['lengths'])
        self.__path = path
        pos = vector(0,0,0)
        self.__target = sphere(pos=pos,radius=0.5,color=color.red)
        self.__target.visible = False
        self.__curr_iter = 0
    
    """    
    function:    def __move_target    
    input parameters: self    
    Notes:     
    """    
    def __move_target(self):
        r = 5
        cz, cy = 0, 5
        angle = np.radians(self.__curr_iter)
        self.__target.pos.x = 10
        self.__target.pos.y = r*np.sin(angle) + cy
        self.__target.pos.z = r*np.cos(angle) + cz
        self.__curr_iter += 10
        if self.__curr_iter > 360:
            self.__curr_iter = 0
        
    """    
    function:    def __random_target    
    input parameters: self    
    Notes:     
    """    
    def __random_target(self):
        pos = vector(0,0,0)
        max_range = int(0.8*self.__total_l )
        pos.x = randint(int(0.1*self.__total_l ),max_range)
        pos.y = randint(int(0.1*self.__total_l ),max_range)
        pos.z = randint(-max_range,max_range)
        self.__target.pos = pos
        
    """    
    function:    def get_target    
    input parameters: self    
    Notes:     
    """    
    def get_target(self):
        if self.__path == 'circle':
            self.__move_target()
        elif self.__path == 'random':
            self.__random_target()
        self.__target.visible = True
        return self.__target
        
"""
function: create_target_path
input parameters: target=None
Notes: 
"""
def create_target_path(target=None):
    pos = vector(0,0,0)
    
    if(target==None):
        target = sphere(pos=pos,radius=0.75,color=color.red)
    else:
        target.pos=pos
        
    return target
"""
Simple gradient descent
"""
def sim_gd_0(model, target_pos):
    #hyperparameters
    steps = 1800
    angles = model['angles']
    reach = model['reach']
    W = np.ones(len(angles))
    X = np.asarray(angles)
    e = 0.0
    total_cost = []
    
    for i in range(len(angles)):
        prev_cost = curr_cost = 0
        for j in range(steps):
            prev_cost = curr_cost
            W0 = np.copy(W)
            W2 = np.copy(W)
            W2[i] += 0.1
            W0[i] -= 0.1
            cost = []
            param_search = [W0, W, W2]
            for param in param_search:
                pred = list(X+param)
                model['angles'] = pred
                reach = update(model)
                cost.append((reach - target_pos).mag)
            min_index = np.argmin(cost)
            curr_cost = cost[min_index]
            total_cost.append(curr_cost)
            W = param_search[min_index]
            if(abs(curr_cost-prev_cost) <= e):
                break
            rate(100)
            
    return total_cost

"""
slightly realistic arm movements?
minimize dist to target at every joint 
"""
def sim_gd_1(model, target_pos):
    #hyperparameters
    steps = 1800
    angles = model['angles']
    W = np.ones(len(angles))
    X = np.asarray(angles)
    e = 0.0
    total_cost = []
    for i in range(len(angles)):
        prev_cost = curr_cost = 0
        for j in range(steps):
            prev_cost = curr_cost
            W0 = np.copy(W)
            W2 = np.copy(W)
            W2[i] += 0.1
            W0[i] -= 0.1
            cost = []
            param_search = [W0, W, W2]
            for param in param_search:
                pred = list(X+param)
                model['angles'] = pred
                reach = update(model)
                if i+1 < len(angles): 
                    reach = model['joint'][i+1].pos
                cost.append((reach - target_pos).mag)
            min_index = np.argmin(cost)
            curr_cost = cost[min_index]
            total_cost.append(curr_cost)
            W = param_search[min_index]
            if(abs(curr_cost-prev_cost) <= e):
                break
            rate(100) 
    return total_cost
"""
simple training function
"""
def train(save=False):
    lengths= [3,6,5,4]
    create_scene()
    model = init_model(lengths)
    model.update({'reach':vector(0,0,0)})
    target = generate_random_target(lengths)
    data = Dataset()
    
    for i in range(100):
        target = generate_random_target(lengths,target)
        #total_cost = sim_gd_0(model,target.pos)
        total_cost = sim_gd_1(model,target.pos)
        data.update_dataset(model,target.pos)
        
    if save:
        data.save_dataset()
    plt.plot(range(len(total_cost)), total_cost)
    plt.ylabel('Distance to target')
    plt.show()
    return


"""
Create a dataset using random angles
"""
def generate_random_angles(model):
    limits = model['limits']
    angles = model['angles']
    for i in range(len(angles)):
        angles[i] = randint(limits['l'][i],limits['u'][i])
        
    return angles

"""
function: generate_dataset
input parameters: size=None
Notes: 
"""
def generate_dataset(size=None):
    if size==None:
        size = 5000
    create_scene(400)
    model = init_model()
    model.update({'reach':vector(0,0,0)})
    data = Dataset()
    for i in range(size):
        model['angles'] = generate_random_angles(model)
        reach = update(model)
        data.update_dataset(model,reach)
    data.save_dataset()    
    return

"""
visualization
"""

"""
function: vis
input parameters: cam_enable=False,com_enable=False
Notes: 
"""
def vis(cam_enable=False,com_enable=False):
    create_scene(400)
    model = init_model()
    model.update({'reach':vector(0,0,0)})
    arduino = Transmission(port='COM5')
    gui = GUI(model)
    gui.build()
    if com_enable:
        com = com_vis(model)
    if cam_enable:
        cam_target = Obj_tracking(model)
    found = False
    while(True):
        if(gui.read_state()):
#            model = rand_move_model(model)
            if com_enable:            
                com = com_vis(model,com)
            if cam_enable:
                _ = cam_target.track(found,True)
                
            model['angles'], model['claw_state'] = gui.read_values()
            data_packet = model['angles'] + [model['claw_state']]
            arduino.transmit(data_packet)
            model['reach'] = update(model)
            
                
        if cam_enable:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                found = True
            elif key == ord('r'):
                found = False
                cam_target.reset()
        
        rate(200)
        
#    cam_target.terminate()
    return

"""
using nn model
"""
def normalize_lengths(X):
    arm_param = {'x_span':(3,15),'y_span':(-6,18),'z_span':(-15,15)}
    X[:,0] = (X[:,0]-arm_param['x_span'][0])/(arm_param['x_span'][1]-arm_param['x_span'][0])
    X[:,1] = (X[:,1]-arm_param['x_span'][0])/(arm_param['x_span'][1]-arm_param['x_span'][0])
    X[:,2] = (X[:,2]-arm_param['x_span'][0])/(arm_param['x_span'][1]-arm_param['x_span'][0])
    return X

"""
function: transform
input parameters: X
Notes: 
"""
def transform(X):
    r = np.linalg.norm(X)
    theta = (np.pi/2) - np.arccos(X[1]/r)
    phi = np.arctan2(X[2],X[0])
    return np.array([r,theta,phi])

"""
function: nn_prediction
input parameters: None
Notes: 
"""
def nn_prediction():
    from keras.models import load_model
    
    error = []
    create_scene(400)
    model = init_model()
    model.update({'reach':vector(0,0,0)})
    nn = load_model('model.hd5')
    target_gen = TargetGenerator(model,'circle')
    print(nn.summary())
    for i in range(1000):
        target = target_gen.get_target()        

        X = np.array([target.pos.x,target.pos.y,target.pos.z])
#        X = transform(X)
#        o = 0
#        X = np.append(X,-np.pi/10)
        X.shape = (1,3)
        X = normalize_lengths(X)
        
        angles = np.degrees(nn.predict(X)[0])
        
        model['angles'] = angles.tolist()[0]
        reach = update(model)
        error.append((target.pos - reach).mag)
        rate(5)
    
    print(np.mean(error))
    return

"""
TO DO
http://www.academia.edu/9165706/Forward_and_inverse_Kinematics_complete_solutions_3DOF_good_reference_for_CrustCrawler_Smart_Arm_Users_
"""
def analytic_sol(model,target_pos):

    x,z,y = target_pos.x,target_pos.y,target_pos.z
    length = model['lengths']
    l1,l2,l3 = length[0], length[1], length[2]+length[3]
    c3 = ((x**2)+(y**2)+((z-l1)**2)-(l2**2)-(l3**2))/(2*l2*l3)
    print(c3)
    s3 = math.sqrt(1-(c3**2))
    model['angles'][0] = np.arctan2(y,x)
    model['angles'][2] = np.arctan2(s3,c3)
    model['angles'][1] = np.arctan2(z-l1,math.sqrt(x**2+y**2))-np.arctan2(l3*s3,l2+(l3*c3))
    return

"""
function: analytic_sol_sim
input parameters: None
Notes: 
"""
def analytic_sol_sim():
    
    error = []
    create_scene(400)
    model = init_model()
    model.update({'reach':vector(0,0,0)})
    target = generate_random_target(model['lengths'])
    for i in range(100):
        target = generate_random_target(model['lengths'],target)
        analytic_sol(model, target.pos)
        reach = update(model)
        error.append((target.pos - reach).mag)
        rate(5)
    
    print(np.mean(error))
    return

"""
function: main
input parameters: None
Notes: 
"""
def main():
#    train()
#    vis()
#    analytic_sol_sim()
#    generate_dataset(20000)
    nn_prediction()

if __name__ == '__main__':
    main()