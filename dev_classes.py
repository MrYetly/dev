#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:42:44 2020

@author: ianich
"""
import pandas as pd
import numpy as np
import os
from scipy import stats


class ActionDomain():
    
    def __init__(self, _min, _max, n_actions):
        self.min = float(_min)
        self.max = float(_max)
        self.n = n_actions
        self.step_size = (self.max - self.min)/(self.n - 1)
        
        #Create list of bin partitions
        actions = []
        for i in range(self.n - 1):
            actions.append(self.min + i*self.step_size)
        actions.append(self.max)
        self.actions = actions
        
    def get_action(self, j):
        return self.actions[j]

class StateDomain():
    
    def __init__(self, _min, _max, n_states):
        self.min = float(_min)
        self.max = float(_max)
        self.n = n_states
        self.bin_size = (self.max - self.min)/self.n
        
        #Create list of bin partitions
        partitions = []
        for i in range(self.n):
            partitions.append(self.min + i*self.bin_size)
        partitions.append(self.max)
        self.partitions = partitions
        
    def get_state(self, x):
        #note state bins are [min, max)
        found = False
        for i in range(self.n):
            _min = states.partitions[i]
            _max = states.partitions[i+1]
            if x >= _min and x < _max:
                found = i
        return found
        
        
class Learner():
    
    def __init__(self, act_dom, state_dom, q_init = 10, e = 0.5, state_init = 0):
        self.act_dom = act_dom
        self.state_dom = state_dom
        self.q_init = float(q_init)
        self.e = e
        self.state = state_init
        
        #create state-action propensity array: actions x states
        propensity_array = np.full(
                shape = (self.act_dom.n, self.state_dom.n),
                fill_value = q_init,
        )
        self.prop_array = propensity_array
        
        
    
    def update(self):
        #update propensity array
        pass
    
    def get_pdf(self):
        #get pdf for actions in current state as defined by self.state
        propensities = self.prop_array[:, self.state]
        probabilities = [p/propensities.sum() for p in propensities]
        pdf = stats.rv_discrete(
                values=(list(range(self.act_dom.n)), probabilities)
        )
        return pdf
        
    def act(self):
        pass
    
    
    
prices = ActionDomain(10,100,6)
states = StateDomain(0, 100, 8)
learner = Learner(prices, states)
