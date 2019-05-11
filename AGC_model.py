#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:25:43 2019

@author: ugot
"""
import math

operation = []

def AGC_RNN(current_load, model):
    for i in range(1, len(current_load)):
        Lt = current_load[i]
        Lt1 = model.predict(Lt)
        
    pass

def AGC_control(current_load, predicted_load, bias):
    operation = []
    sys_freqs = []
    #we start with an initial frequency of 50hz
    sys_freq = 50
    for i, x in (current_load, predicted_load):
        sys_freq = sys_freq
        if x > i:
            #if load increases, frequency reduces
            deltapm = -((x - i))/x
            deltaf  = deltapm/bias
            sys_freq -= deltaf
            sys_freqs.append[sys_freq]
            if 50.05>sys_freq<49.05:
                #1 is normal (no controller)
                operation.append[1]
            elif 50.20>sys_freq<49.05:
                #2 is primary control
                operation.append[2]
            elif 51>sys_freq<49.8:
                #3 is secondary control
                operation.append[3]
            elif 51<sys_freq<49:
                #4 is emergency control
                operation.append[4]
            else: operation.append[4]
            
        elif x < i:
            #if load reduces, frequency increases
            deltapm = -((x - i))/x
            deltaf  = deltapm/bias
            sys_freq += deltaf
            sys_freqs.append[sys_freq]            
            if 50.05>sys_freq<49.05:
                #1 is normal (no controller)
                operation.append[1]
            elif 50.20>sys_freq<49.05:
                #2 is primary control
                operation.append[2]
            elif 51>sys_freq<49.8:
                #3 is secondary control
                operation.append[3]
            elif 51<sys_freq<49:
                #4 is emergency control
                operation.append[4]
            else: operation.append[4]
            
        else:
            #if load stays the same, frequency stays same
            sys_freqs.append[sys_freq]            
            if 50.05>sys_freq<49.05:
                #1 is normal (no controller)
                operation.append[1]
            elif 50.20>sys_freq<49.05:
                #2 is primary control
                operation.append[2]
            elif 51>sys_freq<49.8:
                #3 is secondary control
                operation.append[3]
            elif 51<sys_freq<49:
                #4 is emergency control
                operation.append[4]
            else: operation.append[4]
    return operation, sys_freqs


            
def Give_operation(load, bias):
    operation = []
    sys_freqs = []
    #we start with an initial frequency of 50hz
    firstIteration = True
    for i in range(len(load)):
        if firstIteration == True:
            sys_freq = 50
            sys_freqs.append(50)            
            firstIteration = False
        else:                
            if load[i] > load[i-1]:
                print("load increased")
                #if load increases, frequency reduces
                deltapm = -((load[i] - load[i-1]))/load[i]
                deltaf  = deltapm/bias
                deltaf = deltaf * 50
                sys_freq = 50 - abs(deltaf)
                print(sys_freq)
                sys_freqs.append(sys_freq)            
                if 50.05>sys_freq<49.05:
                    #1 is normal (no controller)
                    operation.append(1)
                elif 50.20>sys_freq<49.05:
                    #2 is primary control
                    operation.append(2)
                elif 51>sys_freq<49.8:
                    #3 is secondary control
                    operation.append(3)
                elif 51<sys_freq<49:
                    #4 is emergency control
                    operation.append(4)
                else: operation.append(4)
                
            elif load[i] < load[i-1]:
                print("load decreased")
                #if load reduces, frequency increases
                deltapm = -((load[i] - load[i-1]))/load[i]
                deltaf  = deltapm/bias
                deltaf = deltaf * 50            
                sys_freq = 50 + abs(deltaf)
                print(sys_freq)
                sys_freqs.append(sys_freq)           
                if 50.05>sys_freq<49.05:
                    #1 is normal (no controller)
                    operation.append(1)
                elif 50.20>sys_freq<49.05:
                    #2 is primary control
                    operation.append(2)
                elif 51>sys_freq<49.8:
                    #3 is secondary control
                    operation.append(3)
                elif 51<sys_freq<49:
                    #4 is emergency control
                    operation.append(4)
                else: operation.append(4)
                
            elif(load[i] == load[i-1]):
                print("load the same")
                print(sys_freq)
                #if load stays the same, frequency stays same
                sys_freqs.append(sys_freq)            
                if 50.05>sys_freq<49.05:
                    #1 is normal (no controller)
                    operation.append(1)
                elif 50.20>sys_freq<49.05:
                    #2 is primary control
                    operation.append(2)
                elif 51>sys_freq<49.8:
                    #3 is secondary control
                    operation.append(3)
                elif 51<sys_freq<49:
                    #4 is emergency control
                    operation.append(4)
                else: operation.append(4)
    return operation, sys_freqs            



                        
                
            
            
    