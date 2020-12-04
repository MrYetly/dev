# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from dev_classes import ActionDomain, StateDomain, Learner
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



### Create Sim Environment ####################################################

#Set parameters
n_o = 500000
n_f = 900000
c_o = 1000000
a_o = c_o - n_f + n_o
b = 1
delta = 1/30000
a_f_max = a_o
a_f_min = 50000
p_o = 100
tot_a_f = 100
decay_min = 1/1000
decay_max = 1/100000
tot_decays = 100
t_max = 10000
t_min = 0
tot_states = 1
e = 0.0
decay = 0.02
q_init = 99990000
update_method = 'bump_else'

#Create action domain of price choices

a_fs = ActionDomain(a_f_min, a_f_max, tot_a_f)
decays = ActionDomain(decay_min, decay_max, tot_decays)

#Create state domain based on number of customers present

states = StateDomain(t_min, t_max, tot_states)

#Create learners for each LSE
a_f_learner = Learner(
        a_fs,
        states,
        e = e,
        q_init = q_init,
        decay = decay,
        update_method = update_method,
)

decay_learner = Learner(
        decays,
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
def generate_price(decay, a_f, t):
    return (a_f - (a_f - a_o)*np.exp(-decay*t))/(2*b)

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
        a_f_learner.set_state(t)
        a = a_f_learner.get_action()
        
        #slope learner acts
        decay_learner.set_state(t)
        d = decay_learner.get_action()
        
        #generate choice, record and compare
        last_price = generate_price(d, a, t)
        chosen.append(last_price)
        optimal.append(opt_p(t))
        
        #calculate and distribute reward
        reward = rev(last_price, t)
        a_f_learner.update(reward)
        decay_learner.update(reward)
        if t%1000 == 0:
            print(t)
    res_df = pd.DataFrame({'chosen': chosen, 'optimal': optimal,})

fig, ax = plt.subplots(figsize = (12,6))    

ax.scatter(x = res_df.index, y = res_df.chosen, s = 1)
ax.scatter(x = res_df.index, y = res_df.optimal, s = 1, color = 'r')

ax.set_ylabel('price')
ax.set_xlabel('timestep')
ax.set_title('Point & Slope learning over 10 time states with trend')

#fig.savefig('indep_ten_states_with_trend.png')


#pdf introspection

data = decay_learner.get_prob_array()
 
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

#fig.savefig('indep_ten_states_with_trend_slope_pdf.png')

