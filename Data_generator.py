 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri March  8 01:14:23 2019

@author: ugot
"""
import random 

def frequency(l1, l2, bias, sys_freq, deltaf):
    ''' Calculate the frequncy deviation between two loads'''
    if l2 > l1:
        #if load increases, frequency reduces
        deltapm = -((l2 - l1))/l2
        deltaf  = deltapm/bias
        deltaf = deltaf * 50
        sys_freq = 50 - abs(deltaf)
        return sys_freq, deltaf

    elif l2 < l1:
        #if load reduces, frequency increases
        deltapm = -((l2 - l1))/l2
        deltaf  = deltapm/bias
        deltaf = deltaf * 50            
        sys_freq = 50 + abs(deltaf)
        return sys_freq, deltaf
    
    elif l2 == l1:
        #if load stays the same
        return sys_freq, deltaf

#131,400 hours
import csv
#from random import gauss
save_dir = 'results'
logfile = open('dataset4.csv', 'w')
logwriter = csv.DictWriter(logfile, fieldnames=['month', 'week','day','hour', 'load', 'deltafrequency', 'sysFreq'])
logwriter.writeheader()
bias = 69.2
sysFreq = 50
deltaf = 0
load1 = 101
firstIteration = True
while(True):
    for month in range(1,13):            
        for week in range(1,5):        
            for day in range(1,8):
                high_load = random.randint(1,18)
                hours = random.randint(12,19)
                print(hours)
                for hour in range(1,hours):
                    if firstIteration == True:
                        logdict = dict(month=month, week=week,day=day,hour=hour,load=load1, deltafrequency=deltaf, 
                                       sysFreq=sysFreq)
                        logwriter.writerow(logdict)     
                        firstIteration = False
                    else:                    
                        if hour <= 6:
                            #Afternoon
                            load2 = random.randint(102, 104)
                            sysFreq, deltaf = frequency(load1, load2, bias, sysFreq, deltaf)
                            logdict = dict(month=month, week=week,day=day,hour=hour,load=load2, deltafrequency=deltaf, 
                                           sysFreq=sysFreq)
                            logwriter.writerow(logdict)
                            load1 = load2
                        elif hour <= 6 and hour==high_load:
                            #Afternoon high load
                            load2 = random.randint(170, 180)
                            sysFreq, deltaf = frequency(load1, load2, bias, sysFreq, deltaf)
                            logdict = dict(month=month, week=week,day=day,hour=hour,load=load2, deltafrequency=deltaf, 
                                           sysFreq=sysFreq)
                            logwriter.writerow(logdict)
                            load1 = load2
                        elif hour > 6 and hour==high_load:
                            #Evening high load
                            load2 = random.randint(200, 256)
                            sysFreq, deltaf = frequency(load1, load2, bias, sysFreq, deltaf)
                            logdict = dict(month=month, week=week,day=day,hour=hour,load=load2, deltafrequency=deltaf, 
                                           sysFreq=sysFreq)
                            logwriter.writerow(logdict)
                            load1 = load2
                        else:
                            #Evening
                            load2 = random.randint(182, 184)
                            sysFreq, deltaf = frequency(load1, load2, bias, sysFreq, deltaf)
                            logdict = dict(month=month, week=week,day=day,hour=hour,load=load2, deltafrequency=deltaf, 
                                           sysFreq=sysFreq)
                            logwriter.writerow(logdict)
                            load1 = load2
        print("End of month", month)
    logfile.close()
    break


#Generate for 3 years
save_dir = 'results'
logfile = open('dataset3.csv', 'w')
logwriter = csv.DictWriter(logfile, fieldnames=['year','month', 'week','day','hour', 'load', 'deltafrequency', 'sysFreq'])
logwriter.writeheader()
bias = 69.2
sysFreq = 50
deltaf = 0
load1 = 101
firstIteration = True
while(True):
    for year in range(1,4):        
        for month in range(1,13):            
            for week in range(1,5):        
                for day in range(1,8):
                    hours = random.randint(12,19)
                    print(hours)
                    for hour in range(1,hours):
                        if firstIteration == True:
                            logdict = dict(year=year,month=month, week=week,day=day,hour=hour,load=load1, deltafrequency=deltaf, 
                                           sysFreq=sysFreq)
                            logwriter.writerow(logdict)     
                            firstIteration = False
                        else:                    
                            if hour <= 6:
                                #Afternoon
                                load2 = random.randint(102, 180)
                                sysFreq, deltaf = frequency(load1, load2, bias, sysFreq, deltaf)
                                logdict = dict(year=year,month=month, week=week,day=day,hour=hour,load=load2, deltafrequency=deltaf, 
                                               sysFreq=sysFreq)
                                logwriter.writerow(logdict)
                                load1 = load2
                            else:
                                #Evening
                                load2 = random.randint(181,256)
                                sysFreq, deltaf = frequency(load1, load2, bias, sysFreq, deltaf)
                                logdict = dict(year=year,month=month, week=week,day=day,hour=hour,load=load2, deltafrequency=deltaf, 
                                               sysFreq=sysFreq)
                                logwriter.writerow(logdict)
                                load1 = load2
            print("End of month", month)
    logfile.close()
    break
