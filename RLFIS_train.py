#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:46:00 2019

@author: cxx
"""
from pompy import models, processors
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, Circle
import time
import random
import math
import robot as Robot
import distribution_functions as df
import fuzzy_levy_para as fp
import _pickle as pickle
import os
import matplotlib.pyplot as plt
from DDPG import DDPG

img_show = 0
sim_region = models.Rectangle(x_min=0., x_max=40., y_min=-10., y_max=10.)
# Set up figure
if img_show:
    fig = plt.figure(figsize=(10, 5))
    plt.ion()
    ax = fig.add_axes([0., 0., 1., 1.])
    plt.tick_params(labelsize=18)
    
    plt.xlabel("x (m)",size=18)
    plt.ylabel("y (m)",size=18)

robot = Robot.RobotModel(x = 25., y = 2., heading = math.pi, arrow_length = 0.05, radius = 0.08, v_left = 1, v_right = 1)

seed = int(time.time())#20180517
rng = np.random.RandomState(seed)
step_limit = 60
l_min = 0.5
miu = 2.8
gamma= 0.05
con_p_a = 0.
con_p_f = 0
#con_c = 0.
theta = 0.8
miu_min = 1.01
miu_max = 3.0
gamma_min = 0.
gamma_max = 1.

Var = 1           #initial epsilon ie. exploration rate
VarDecayRate = 0.0001   #by how much we decrease epsilon each epoch 0.0002
minVar = 0.1            #minimum of epsiol 0.001

MaxEpisodes = 3000        #on how many epochs model is trained on
learningRate = 0.0001   #learning rate of the model
gamma = 0.9             #gamma parameter which defines by how much next state influences previous state
batchSize = 32          #size of batch neural network is trained on every iteration (every move) 
MemoryCapacticy = 5000

defaultReward = -1    #the default reward for every action
negativeReward = -10     #the reward for hitting itself or wall
positiveReward = 20      #the reward for finding the source


s_dim = 2
a_dim = 3
a_low = np.array([1.02,0.,0.])
a_high = np.array([3.,1.,1.])
# s_low = np.array([0.,-20,0.,-10.])
# s_high = np.array([20.,20,40.,10.])
s_low = np.array([0.,-20.])
s_high = np.array([20.,20.])

Replacement = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter=10)
][0]            # you can try different target replacement strategies

ddpg = DDPG(state_dim=s_dim,
            action_dim=a_dim,
            a_low=a_low,
            a_high=a_high,
            s_low = s_low,
            s_high = s_high,
            replacement=Replacement,
            memory_capacticy=MemoryCapacticy,
            train = True)

# Define concentration array (image) generator parameters
array_gen_params = {
    'array_z': 0.,
    'n_x': 40*10,
    'n_y': 20*10,
    'puff_mol_amount': 8.3e8#8.3e8
}

# Create concentration array generator object
array_gen = processors.ConcentrationArrayGenerator(
    array_xy_region=sim_region, **array_gen_params)

start = True

#f=open('reward','w+')
        
# Define animation update function
def env_update(s,t):
    global conc_array
    global wind_model
    global con_p_f
    global Var
    
    a, factor = RLFIS_levy_taxis(s)
    if (robot.x>=39.95)|(robot.x<0)|(robot.y>=9.95)|(robot.y<=-9.95) :
        state = 1
        r = negativeReward
        return a, [],r,state
        
    
    with open("environment_models/{}-plume.pkl".format(t), 'rb') as f:
        plume_model = pickle.load(f)   # read file and build object
    with open("environment_models/{}-wind.pkl".format(t), 'rb') as f:
        wind_model = pickle.load(f)   # read file and build object

    conc_array = array_gen.generate_single_array(plume_model.puff_array)
    con_c = robot.get_concentration(conc_array.T)
    #print(con_c)
    delta_c = con_c - con_p_f
    con_p_f = con_c
    s_ = np.array([con_c,delta_c])
    
    if img_show:
        plt.cla()
        ax = fig.add_axes([0., 0., 1., 1.])
        plt.tick_params(labelsize=18)
    
        plt.xlabel("x (m)",size=18)
        plt.ylabel("y (m)",size=18)
    
        conc_im = ax.imshow(
            conc_array.T, extent=sim_region, vmin=0., vmax=1e10, cmap='Reds')
    
#    adaptive_levy_taxis()
    if img_show:
        ax.add_patch(Circle(xy = (5, 0), radius=2, alpha=0.5,fc='y'))
        ax.add_patch(Circle(xy=(robot.x, robot.y), radius=robot.radius, alpha=0.5,fc='g'))
        ax.arrow(robot.x, robot.y, math.cos(robot.heading)*robot.arrow_length, math.sin(robot.heading)*robot.arrow_length, head_width=0.05, head_length=0.1, fc='g', ec='g')
        ax.plot(robot.trajectory[:,0],robot.trajectory[:,1],linewidth=2)
        #show wind
        for i in range(10):
            for j in range(-3,3):
                vx,vy = wind_model.velocity_at_pos(i*5,-j*5)
                wind_angle = math.atan2(-vy,vx)
                wind_v = math.sqrt(vx**2+vy**2)
                ax.arrow(i*5, j*5, vx,-vy,head_width=0.2, head_length=0.5, fc='k', ec='k')
    
        plt.draw()
        plt.pause(0.0001)
    
    if ((robot.x-5)**2+robot.y**2<4.):
        state = 2
        r = positiveReward
    else:
        state = 0
        r = defaultReward+con_c/30.*factor
    
    return a, s_,r,state

        
def RLFIS_levy_taxis(s):
    
    a = ddpg.choose_action(s.reshape(1,-1)).reshape(a_dim,)
    a = np.clip(np.random.normal(a,Var),a_low,a_high)
    [miu, gamma, bias] = a
    ran = random.random()
    m_l = l_min*math.pow(ran,1./(1-miu))
    if (m_l>1):
        m_l = 1

    vx, vy = wind_model.velocity_at_pos(robot.x, -robot.y)
    wind_angle = math.atan2(-vy,vx )
    #print(vx, -vy, wind_angle)
    ran = random.random()
    val = 2 * math.atan((1.0 - gamma) / (1.0 + gamma) * math.tan(np.pi * (ran - 0.5)))
    t_a = val+ bias*(wind_angle+math.pi)+(1.-bias)*robot.heading-robot.heading


    robot.turn_clockwise(t_a)
    robot.move_length(m_l)
    robot.step = robot.step + 1
    
    return a, m_l * math.cos(robot.heading-wind_angle-math.pi)
    
def env_init():
    global conc_array
    global wind_model
    global con_p_f
    
    
    with open("environment_models/0-plume.pkl", 'rb') as f:
        plume_model = pickle.load(f)   # read file and build object
    with open("environment_models/0-wind.pkl", 'rb') as f:
        wind_model = pickle.load(f)   # read file and build object
    conc_array = array_gen.generate_single_array(plume_model.puff_array)

    con_c = robot.get_concentration(conc_array.T)
    #print(con_c)
    delta_c = con_c - con_p_f
    con_p_f = con_c
    #s = np.array([con_c,delta_c,robot.x,robot.y])
    s = np.array([con_c,delta_c])
    return s

mean_reward = 0
model_save = 1
results = []
f_r=open('reward','w+')
for i in range(MaxEpisodes):
    robot = Robot.RobotModel(x = 23+4*random.random(), y = -2+4*random.random(), heading = math.pi, arrow_length = 0.05, radius = 0.08, v_left = 1, v_right = 1)
    s = env_init()
    ep_reward = 0
    for step in range(1,step_limit):
        a, s_, r, state =  env_update(s,step)
        if s_!= []:
            ddpg.store_transition(s, a, r, s_)
    
        if (state != 1) & (ddpg.pointer > 3*batchSize):
            if Var > minVar:
                Var -= VarDecayRate    # decay the action randomness
            ddpg.learn()
        
        s = s_
        ep_reward += r
        if state:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % Var, )
            mean_reward += ep_reward
            break
        if step == step_limit-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % Var, )
            mean_reward += ep_reward
    if (i> 0) & (i%20 == 0):
        mean_reward /= 20.
        results.append(mean_reward)
        f_r.write(str(mean_reward)+'\n')
        f_r.flush()
        plt.plot(results)
        plt.ylabel('Average Score')
        plt.xlabel('Epoch / 20')
        plt.show()
        print(mean_reward)
        mean_reward = 0
        if model_save & (i%20 == 0):
            ddpg.save_actor('actor_models/actor_episode_{}.pkl'.format(i))
            
        
        
        