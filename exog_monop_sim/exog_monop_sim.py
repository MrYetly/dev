# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from dev_classes import ActionDomain, StateDomain, Learner
import matplotlib.pyplot as plt


### Create Sim Environment ####################################################

#Set parameters
p_max = 149
p_min = 50
p_o = 100
tot_prices = 100
n_max = 1000000
n_min = 0
#n_o = 0
#n_f = 500000
n_o = 500000
n_f = 900000
c_o = n_max
b = 1
delta = 1/10000
tot_states = 1
e = 0.1
decay = 0.02
q_init = 99990000
update_method = 'bump_else'

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
def rev(p, t):
    return p*c_o - ((n_f - n_o*np.exp(-delta*t))/p_o + b)*p**2

#find optimal price to check results
def opt_p(t):
    return (p_o*c_o)/(2*(n_f - n_o*np.exp(-delta*t) + p_o*b))

#find number of solar customers for a given price
def get_solar_cust(p, t):
    return min(n_max-1, (n_f - n_o*np.exp(-delta*t))*(p/p_o))

#find optimal state to check results
def opt_state(t):
    return states.get_state(get_solar_cust(opt_p(t), t))

### Run Simulation ############################################################

runs = 1

res_df = pd.DataFrame()
for j in range(runs):
    timesteps = int(round(1/delta))
    p_last = p_o
    optimal = []
    chosen = []
    print(j)
    for t in range(timesteps):
        learner.set_state(get_solar_cust(p_last, t))
        p_last = learner.get_action()
        chosen.append(p_last)
        optimal.append(opt_p(t))
        reward = rev(p_last, t)
        learner.update(reward)
        if t%1000 == 0:
            print(t)
    res_df = pd.DataFrame({'chosen': chosen, 'optimal': optimal,})

fig, ax = plt.subplots(figsize = (12,6))    

ax.scatter(x = res_df.index, y = res_df.chosen, s = 1)
ax.scatter(x = res_df.index, y = res_df.optimal, s = 1, color = 'r')

ax.set_ylabel('price')
ax.set_xlabel('timestep')
ax.set_title('10 states with trend')

#fig.savefig('ten_states_with_trend.png')
