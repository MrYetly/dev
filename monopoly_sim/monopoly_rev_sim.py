# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from dev_classes import ActionDomain, StateDomain, Learner
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


### Create Sim Environment ####################################################

#Set parameters
p_max = 149
p_min = 50
p_o = 100
tot_prices = 100
n_max = 1000000
n_min = 0
n_o = 500000
c_o = n_max
b = 1
tot_states = 10
e = 0.1
decay = 0.02
q_init = 9999
update_method = 'share'

#Create action domain of price choices

prices = ActionDomain(p_min, p_max, tot_prices)

#Create state domain based on number of customers present

states = StateDomain(n_min, n_max, tot_states)

#Create learners for each LSE
learner = Learner(
        prices,
        states,
        e = e,
        q_init = q_init,
        decay = decay,
        update_method = update_method,
)

#Create revenue function to serve as reward
def rev(p):
    return p*c_o - (n_o/p_o + b)*p**2

#find optimal price to check results
def opt_p():
    return (p_o*c_o)/(2*(n_o + p_o*b))

#find number of solar customers for a given price
def get_solar_cust(p):
    return n_o*(1+(p-p_o)/p_o)

#find optimal state to check results
def opt_state():
    return states.get_state(get_solar_cust(opt_p()))

### Run Simulation ############################################################

runs = 1
results = []
res_df = pd.DataFrame()
for j in range(runs):
    timesteps = 10000
    p_last = p_o
    print(j)
    for i in range(timesteps):
        learner.set_state(get_solar_cust(p_last))
        p_last = learner.get_action()
        reward = rev(p_last)
        learner.update(reward)
        if i%100 == 0:
            print(i)
    r = learner.get_pdf().expect()
    r = p_min + r*(p_max-p_min)/(tot_prices-1)
    results.append(r)

#res_df[f'{tot_states}_states_{q_init}_q_init'] = results
#res_df.describe().to_latex(f'{tot_states}_states_{q_init}_q_init.tex')


#Graph results

data = learner.get_prob_array()
 
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
ax1.set_ylabel('price index')
ax1.set_xlabel('state index')
ax1.set_zlabel('probability')

plt.show()   
fig.savefig(f'monop_{tot_states}_states_{update_method}.png')


#print expectation of price choice and final state
r = learner.get_pdf().expect()
r = p_min + r*(p_max-p_min)/(tot_prices-1)

print(f'expected price choice: {r}, optimal: {opt_p()}')
print(f'last state: {learner.state}, optimal: {opt_state()}')
