# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from dev_classes import ActionDomain, StateDomain, Learner
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



### Create Sim Environment ####################################################

#Set parameters
p_max = 125
p_min = 56
p_o = 100
tot_prices = 70
s_min = -0.005205730142903747
s_max = -3.8192170897498975e-05
tot_slopes = 100
t_max = 100000
t_min = 0
n_o = 500000
n_f = 900000
c_o = 1000000
b = 1
delta = 1/30000
tot_states = 10
e = 0.0
decay = 0.02
q_init = 99990000
update_method = 'bump_else'

#Create action domain of price choices

prices = ActionDomain(p_min, p_max, tot_prices)
slopes = ActionDomain(s_min, s_max, tot_slopes)

#Create state domain based on number of customers present

states = StateDomain(t_min, t_max, tot_states)

#Create learners for each LSE
point_learner = Learner(
        prices,
        states,
        e = e,
        q_init = q_init,
        decay = decay,
        update_method = update_method,
)

slope_learner = Learner(
        slopes,
        states,
        e = e,
        q_init = q_init,
        decay = decay,
        update_method = update_method,
)

#Create revenue function to serve as reward
def rev(p, t):
    return p*c_o - ((n_f - n_o*np.exp(-delta*t))/p_o + b)*p**2

#find optimal price to check results
def opt_p(t):
    return (p_o*c_o)/(2*(n_f - n_o*np.exp(-delta*t) + p_o*b))

def opt_slope(t):
    return -(p_o*c_o*delta*n_o*np.exp(-delta*t))/(2*((n_f - n_o*np.exp(-delta*t) + p_o*b))**2)

#find number of solar customers for a given price
def get_solar_cust(p, t):
    return min(c_o-1, (n_f - n_o*np.exp(-delta*t))*(p/p_o))

#generate price choice from point and slope choices
def generate_price(s, p, t):
    t_o = states.partitions[states.get_state(t) + 1]
    return p + s*(t-t_o)

#find optimal state to check results
def opt_state(t):
    return states.get_state(get_solar_cust(opt_p(t), t))

### Run Simulation ############################################################

runs = 1

res_df = pd.DataFrame()
for j in range(runs):
    timesteps = t_max-1
    optimal = []
    chosen = []
    print(j)
    for t in range(timesteps):
        
        #point learner acts
        point_learner.set_state(t)
        p = point_learner.get_action()
        
        #slope learner acts
        slope_learner.set_state(t)
        s = slope_learner.get_action()
        
        #generate choice, record and compare
        last_price = generate_price(s, p, t)
        chosen.append(last_price)
        optimal.append(opt_p(t))
        
        #calculate and distribute reward
        reward = rev(last_price, t)
        point_learner.update(reward)
        slope_learner.update(reward)
        if t%1000 == 0:
            print(t)
    res_df = pd.DataFrame({'chosen': chosen, 'optimal': optimal,})

fig, ax = plt.subplots(figsize = (12,6))    

ax.scatter(x = res_df.index, y = res_df.chosen, s = 1)
ax.scatter(x = res_df.index, y = res_df.optimal, s = 1, color = 'r')

ax.set_ylabel('price')
ax.set_xlabel('timestep')
ax.set_title('Point & Slope learning over 10 time states with trend')

fig.savefig('indep_ten_states_with_trend.png')


#pdf introspection

data = slope_learner.get_prob_array()
 
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111, projection='3d')
       
_x = np.arange(data.shape[1])
_y = np.arange(data.shape[0])
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = data[y, x]
bottom = np.zeros_like(top)
width = depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
ax1.set_title('pdf')
ax1.set_ylabel('action index')
ax1.set_xlabel('state index')
ax1.set_zlabel('probability')

plt.show() 

fig.savefig('indep_ten_states_with_trend_slope_pdf.png')

