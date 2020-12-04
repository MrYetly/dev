# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from dev_classes import ActionDomain, StateDomain, Learner
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('../Simulation Results/demand.csv')

#ex_data = data[['N 1', 'D 1']]
#ex_data.head().to_html('ex_data')


### Create Sim Environment ####################################################

#Set parameters
p_max = 108.0
p_min = 70.0
tot_prices = 10
n_max = data[['N 0', 'N 1', 'N 2', 'N 3', 'N 4', 'N 5']].max().max()
n_min = 0
tot_states = 10
e = 0.1
q_init = 100000

#Create action domain of price choices

prices = ActionDomain(p_min, p_max, tot_prices)

#Create state domain based on number of customers present
n_max = data[['N 0', 'N 1', 'N 2', 'N 3', 'N 4', 'N 5']].max().max()
n_min = 0
states = StateDomain(n_min, n_max, tot_states)

#Create learners for each LSE
learners = {
        '1': Learner(prices, states, e = e, q_init = q_init),
}


### run simulation ############################################################

#track days in state
time_in_state = np.zeros(states.n)

#each row in the loop represents an hour passing by
for i, row in data.iterrows():
    if i % 10000 == 0:
        print(i)
    #if not (i-12) % 24 == 0:
        #continue
    #Each hour, each learner observes state, acts, observes a response, and updates
    for k, learner in learners.items():
        learner.set_state(row[f'N {k}'])
        time_in_state[learner.state] += 1
        price = learner.get_action()
        #reward is revenue
        demand = row[f'D {k}']
        reward = price * demand
        learner.update(reward)

#plot results
data = learners['1'].get_prob_array()
 
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
ax1.set_ylabel('prices')
ax1.set_xlabel('states')
ax1.set_zlabel('probability')

plt.show()
fig.savefig('result.png')
