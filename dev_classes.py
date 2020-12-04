#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:42:44 2020

@author: ianich
"""
import numpy as np
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
            _min = self.partitions[i]
            _max = self.partitions[i+1]
            if x >= _min and x < _max:
                found = i
        return found
        
        
class Learner():
    
    def __init__(self,
                 act_dom,
                 state_dom,
                 q_init = 10,
                 e = 0.5,
                 decay = 0.02,
                 state_init = 0,
                 pdf_method = 'simple',
                 update_method = 'share',
                 ):
        self.act_dom = act_dom
        self.state_dom = state_dom
        self.q_init = float(q_init)
        self.e = e
        self.decay = decay
        self.state = state_init
        self.last_act = None
        self.pdf_method = pdf_method
        self.update_method = update_method
        
        #create state-action propensity array: actions x states
        propensity_array = np.full(
                shape = (self.act_dom.n, self.state_dom.n),
                fill_value = self.q_init,
        )
        self.prop_array = propensity_array
        
        
    def update(self, r, a = None):
        '''
        update action propensities for current state
        '''
        if a == None:
            a = self.last_act
        prop = self.prop_array[:, self.state]
        for i in range(len(prop)):
            #decay all action propensities to prioritize newer learning
            prop[i] = (1 - self.decay)*prop[i]
            #apply response function conditional on reward, selected action
            if self.update_method == 'share':
                if i == a:
                    prop[i] = prop[i] + (1-self.e)*r
                elif i == (a+1) or i == (a-1):
                    prop[i] = prop[i] + self.e*0.5*r
            elif self.update_method == 'bump_else':
                if i == a:
                    prop[i] = prop[i] + (1-self.e)*r
                else:
                    prop[i] = prop[i] + self.e*prop[i]/(self.act_dom.n - 1)
        return self.prop_array
    
    def get_pdf(self, state = None):
        '''
        get pdf for actions in current state as defined by self.state
        '''
        if state == None:
            state = self.state
        prop = self.prop_array[:, state]
        if self.pdf_method == 'GB':
            T = 1
            prob = [np.exp(p/T)/np.exp(prop/T).sum() for p in prop]
        if self.pdf_method == 'simple':
            prob = [p/prop.sum() for p in prop]
        pdf = stats.rv_discrete(
                values=(list(range(self.act_dom.n)), prob)
        )
        return pdf
        
    def get_action(self):
        '''
        sample pdf in current state to choose action
        '''
        a = self.get_pdf().rvs()
        self.last_act = a
        return self.act_dom.get_action(a)
    
    def set_state(self, x):
        '''
        Set current state
        '''
        self.state = self.state_dom.get_state(x)
        
    def get_prob_array(self):
        output = np.array(self.prop_array)
        for i in range(output.shape[1]):
            state = output[:,i]
            state_prob = np.array([p/state.sum() for p in state])
            output[:,i] = state_prob
        return output
    

    
    
'''  
prices = ActionDomain(10,100,6)
states = StateDomain(0, 100, 8)
learner = Learner(prices, states, update_method = 'bump_else')


print(learner.prop_array)
print(learner.state)
learner.set_state(65)
learner.get_action()
print(learner.last_act)
r = 5.3365
test = learner.update(r)
print(learner.prop_array)
'''