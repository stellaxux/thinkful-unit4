# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:25:11 2016

@author: Xin
"""
import random
import matplotlib.pyplot as plt
import numpy as np

class Coin(object):
    '''this is a simple fair coin, can be pseudorandomly flipped'''
    sides = ('heads', 'tails')
    last_result = None

    def flip(self):
        '''call coin.flip() to flip the coin and record it as the last result'''
        self.last_result = result = random.choice(self.sides)
        return result

class normalVariable(object):
    last_result = None
    def generator(self):
        self.last_result = result = np.random.randn()
        return result
# let's create some auxilliary functions to manipulate the coins:

def create_variable(number):
    '''create a list of a number of coin objects'''
    return [normalVariable() for _ in xrange(number)]

def generate(variables):
    '''side effect function, modifies object in place, returns None'''
    for v in variables:
        v.generator()

def maximum(normalvariables):
    return max(normalVariable.last_result for normalVariable in normalvariables)

def minimum(normalvariables):
    return min(normalVariable.last_result for normalVariable in normalvariables)

def trial(normalvariables):
    return [normalVariable.last_result for normalVariable in normalvariables]
    
variables = create_variable(1000)
trials = []
for i in xrange(100):
    generate(variables)
    trials.extend(trial(variables))
    #trials.append(maximum(variables))
    #trials.append(minimum(variables))
    
plt.figure()
plt.hist(trials)
