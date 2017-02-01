# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 19:31:39 2017

@author: Administrator
"""

from __future__ import division
import numpy as np
from numpy.random import randint,rand


def myfunc(son=0,mother=5,mail=1,):
    
  '''
  Parameter :son(int), mother(int), mail(int,float,double)
  Son/mother means the percentage of self-driving cars, mail means path distance.This
  function is used to calculate the number of 200 iterations car that will change lanes,
  the number of cars, the percentage of car changing line under different ratio of selfdriving
  cars, the total flow of cars and average per iteration car traffic.
  Return: List of five elements.The order for the elements is the same as above.
  '''
    
    k1_arr = []
    k2_arr = []   
    flow_arr = []
    
    for x in range(200):
        
        #Initial number of cars
        #to "0" means no car, "1" means a car
        length = int(mail*5280/15)
        car_mat = randint(2,size = (length,2))
        car_mat_copy1 = car_mat.copy()
        car_mat_copy2 = car_mat.copy()
    
        #Initialize the speed of each vehicle
        car_speed = []
        for i in car_mat.ravel():
            if i != 0:
                car_speed.append(randint(1,6,size=1))
            else:
                car_speed.append(0)
        car_speed = np.array(car_speed).reshape((length,2))

        #Initialize the expected speed of each vehicle
        car_mean_speed = []
        for i in car_mat.ravel():
            if i != 0:
                car_mean_speed.append(randint(1,6,size=1))
            else:
                car_mean_speed.append(0)
        car_mean_speed = np.array(car_mean_speed).reshape((length,2))  

        #the index of car on the first and second road
        index1 = [index for index,i in enumerate(car_mat[:,0]) if i == 1]
        index2 = [index for index,i in enumerate(car_mat[:,1]) if i == 1]
        #Use "0" for ordinary cars, "1" indicates that self_driving car
        rate_arr = [1]*son+[0]*(mother-son)
        rate1 = [rate_arr[randint(mother,size = 1)] for i in range(len(index1))]
        rate2 = [rate_arr[randint(mother,size = 1)] for i in range(len(index2))]
        #Create rate and index of dictionary
        index_rate1 = dict(zip(index1,rate1))
        index_rate2 = dict(zip(index2,rate2))
##=============================================================================
        #Calculate the number of vehicles that can move vertically 
        #on the first road
        k1 = 0
        for i in range(1,len(index1)):
            
            # Condition four: rand () < p
            if index_rate1[index1[i]] == 1:
                u = rand(1)
                p = rand(1)
                if u < p:
                    k1 = k1+1
                    car_mat_copy1[index1[i],1] = 1
                    car_mat_copy1[index1[i],0] = 0
                    #Exchange rate
                    v_car = car_speed[index1[i],1]
                    car_speed[index1[i],1] = car_speed[index1[i],0]
                    car_speed[index1[i],0] = v_car
                    
                    v_mean_car = car_mean_speed[index1[i],1]
                    car_mean_speed[index1[i],1] = car_mean_speed[index1[i],0]
                    car_mean_speed[index1[i],0] = v_mean_car
                else: 
                    continue
   
            #Condition three: gapback(tli, xi) ≥ Li
            if index1[i] in index2:
                if index1[i] == 1 and index_rate2[index1[i]] == 1:
                    m = randint(1,size=1)
                    if m == 1:
                        k1 = k1+2
                        car_mat_copy1[index1[i],1] = 1
                        car_mat_copy1[index1[i],0] = 0
                        #Exchange rate
                        v_car = car_speed[index1[i],1]
                        car_speed[index1[i],1] = car_speed[index1[i],0]
                        car_speed[index1[i],0] = v_car
                        
                        v_mean_car = car_mean_speed[index1[i],1]
                        car_mean_speed[index1[i],1] = car_mean_speed[index1[i],0]
                        car_mean_speed[index1[i],0] = v_mean_car
                    else:
                        continue
                else:
                    continue
            
            gap_head = index1[i]-index1[i-1]-1
            if gap_head > car_speed[i,0]:             
                continue
        
            #Condition two: gaphead(tli, xi) > gaphead(li, xi)
            if index2[0] > index1[i]:  
                k1 = k1+1
                car_mat_copy1[index1[i],1] = 1
                car_mat_copy1[index1[i],0] = 0
                #Exchange rate
                v_car = car_speed[index1[i],1]
                car_speed[index1[i],1] = car_speed[index1[i],0]
                car_speed[index1[i],0] = v_car
        
                v_mean_car = car_mean_speed[index1[i],1]
                car_mean_speed[index1[i],1] = car_mean_speed[index1[i],0]
                car_mean_speed[index1[i],0] = v_mean_car 
                continue
    
            #Condition one: gaphead(li, xi) ≤ vi
            #Find the optimal "index2" location
            for j in range(len(index2)):
                if j+1 < len(index2):
                    if  index2[j]<index1[i] and index2[j+1]>index1[i]:
                        gap_head_l = index1[i]-index2[j]-1 
                        if gap_head_l > gap_head:
                            k1 = k1+1
                            car_mat_copy1[index1[i],1] = 1
                            car_mat_copy1[index1[i],0] = 0
                            #Exchange rate
                            v_car = car_speed[index1[i],1]
                            car_speed[index1[i],1] = car_speed[index1[i],0]
                            car_speed[index1[i],0] = v_car
                    
                            v_mean_car = car_mean_speed[index1[i],1]
                            car_mean_speed[index1[i],1] = car_mean_speed[index1[i],0]
                            car_mean_speed[index1[i],0] = v_mean_car
                        else:   continue
                else:
                    gap_head_l = index1[i]-index2[j]-1 
                    if gap_head_l > gap_head:
                        k1 = k1+1
                        car_mat_copy1[index1[i],1] = 1
                        car_mat_copy1[index1[i],0] = 0
                        #Exchange rate
                        v_car = car_speed[index1[i],1]
                        car_speed[index1[i],1] = car_speed[index1[i],0]
                        car_speed[index1[i],0] = v_car
                        
                        v_mean_car = car_mean_speed[index1[i],1]
                        car_mean_speed[index1[i],1] = car_mean_speed[index1[i],0]
                        car_mean_speed[index1[i],0] = v_mean_car
        
        k1_arr.append(k1)
##=============================================================================
        #Calculate the number of vehicles that can move vertically 
        #on the sceond road
        k2 = 0
        for i in range(1,len(index2)):
            
            # Condition four: rand () < p
            if index_rate2[index2[i]] == 1:
                u = rand(1)
                p = rand(1)
                if u < p:
                    k2 = k2+1
                    car_mat_copy2[index2[i],1] = 1
                    car_mat_copy2[index2[i],0] = 0
                    #Exchange rate
                    v_car = car_speed[index2[i],1]
                    car_speed[index2[i],1] = car_speed[index2[i],0]
                    car_speed[index2[i],0] = v_car
                    
                    v_mean_car = car_mean_speed[index2[i],1]
                    car_mean_speed[index2[i],1] = car_mean_speed[index2[i],0]
                    car_mean_speed[index2[i],0] = v_mean_car
                else: 
                    continue            

            
            if index2[i] in index1: continue  
    
            gap_head = index2[i]-index2[i-1]-1
            if gap_head > car_speed[i,0]:             
                continue
            
             #Condition two: gaphead(tli, xi) > gaphead(li, xi)
            if index1[0] > index2[i]:  
                k2 = k2+1
                car_mat_copy2[index2[i],1] = 1
                car_mat_copy2[index2[i],0] = 0
                #Exchange rate
                v_car = car_speed[index2[i],1]
                car_speed[index2[i],1] = car_speed[index2[i],0]
                car_speed[index2[i],0] = v_car
                
                v_mean_car = car_mean_speed[index2[i],1]
                car_mean_speed[index2[i],1] = car_mean_speed[index2[i],0]
                car_mean_speed[index2[i],0] = v_mean_car
                continue
            
            #Condition one: gaphead(li, xi) ≤ vi
            #Find the optimal "index1" location
            for j in range(len(index1)):
                if j+1 < len(index1):
                    if  index1[j]<index2[i] and index1[j+1]>index2[i]:
                        gap_head_l = index2[i]-index1[j]-1 
                        if gap_head_l > gap_head:
                            k2 = k2+1
                            car_mat_copy2[index2[i],1] = 1
                            car_mat_copy2[index2[i],0] = 0
                            #Exchange rate
                            v_car = car_speed[index2[i],1]
                            car_speed[index2[i],1] = car_speed[index2[i],0]
                            car_speed[index2[i],0] = v_car
                            
                            v_mean_car = car_mean_speed[index2[i],1]
                            car_mean_speed[index2[i],1] = car_mean_speed[index2[i],0]
                            car_mean_speed[index2[i],0] = v_mean_car
                        else:   continue
                else:
                    gap_head_l = index2[i]-index1[j]-1 
                    if gap_head_l > gap_head:
                        k2 = k2+1
                        car_mat_copy2[index2[i],1] = 1
                        car_mat_copy2[index2[i],0] = 0
                        #Exchange rate
                        v_car = car_speed[index2[i],1]
                        car_speed[index2[i],1] = car_speed[index2[i],0]
                        car_speed[index2[i],0] = v_car
                        
                        v_mean_car = car_mean_speed[index2[i],1]
                        car_mean_speed[index2[i],1] = car_mean_speed[index2[i],0]
                        car_mean_speed[index2[i],0] = v_mean_car
                        
        k2_arr.append(k2)
                        
        #Combining two matrices 
                        
        df = np.zeros((len(car_mat),2))
        df[:,0] =car_mat_copy2[:,0]
        df[:,1] =car_mat_copy1[:,1]
        car_speed = car_speed.ravel()
        df = df.ravel()
        for i in range(len(car_speed)):
            if car_speed[i] == 0:
                df[i] = 0
            else:   continue
        df = df.reshape(len(car_mat),2)
        car_speed = car_speed.reshape(len(car_mat),2)
        #print k1,k2
#####=======================================================================
#####=======================================================================
        '''Longitudinal movement'''
        ####Calculate the "gap" distance of the first road
        gap_dis1 = []
        for index,i in enumerate(df[:,0]):
            if index == 0: 
                gap_dis1.append(0)
                continue
            if i == 0:  gap_dis1.append(0)
            else:   gap_dis1.append(index)    

        keys = np.array(gap_dis1)[np.array(gap_dis1)!= 0]
        items = range(len(keys))
        dict1 = dict(zip(keys,items))
        #Swap "key" and "items""
        dict2 = {value:key for key, value in dict1.items()}

        #Fill value of "gap1"
        gap1 = []
        for i in gap_dis1:
            if i==0: gap1.append(0)
            else:
                try:
                    key = dict1[i]
                    item = dict2[key+1]
                    gap1.append(item-i)
                except:
                    gap1.append(0)

        ####Calculate the "gap" distance of the second road
        gap_dis2 = []
        for index,i in enumerate(df[:,1]):
            if index == 0: 
                gap_dis2.append(0)
                continue
            if i == 0:  gap_dis2.append(0)
            else:   gap_dis2.append(index)    

        keys = np.array(gap_dis2)[np.array(gap_dis2)!= 0]
        items = range(len(keys))
        dict1 = dict(zip(keys,items))
        #Swap "key" and "items""
        dict2 = {value:key for key, value in dict1.items()}
        
        #Fill value of "gap2"
        gap2 = []
        for i in gap_dis2:
            if i==0: gap2.append(0)
            else:
                try:
                    key = dict1[i]
                    item = dict2[key+1]
                    gap2.append(item-i)
                except:
                    gap2.append(0)
        #######Get the final distance of "gap"
        gap_dis = np.zeros((len(car_mat),2))
        gap_dis[:,0] = gap1
        gap_dis[:,1] = gap2

        #######Calculate the longitudinal velocity
        '''
        In order to facilitate, I put the two-dimensional array into
        a one-dimensional array
        '''
        car_speed = car_speed.ravel(order = "F")
        car_mean_speed = car_mean_speed.ravel(order = "F")
        gap_dis = gap_dis.ravel(order = "F")
        df = df.ravel(order = "F")
        combine_rate = rate1+rate2
        v_final = []
        
        for i in range(len(df)):
            kk = 0
            if df[i] == 0:  v_final.append(0)
            else:
                v = min(car_speed[i]+1,car_mean_speed[i])
                v = min(v,gap_dis[i])
                if combine_rate[kk] == 1:
                    v_final.append(int(v))
                    kk = kk+1
                    continue
                if v<car_speed[i]:
                    aa = rand(1)
                    bb = rand(1)
                    if bb<aa:
                        v = max(v-1,0)
                        v_final.append(int(v))
                    else: v_final.append(v)
                else:
                    v_final.append(v)
        v_final = np.array(v_final).reshape((2,len(car_mat))).T

        #########Calculate the number of vehicles after each iteration

        #Divided into two groups
        v_1 = v_final[:,0]
        v_2 = v_final[:,1]
        
        #The number of vehicles on the changes in the first road
        flow = 0
        for index,i in enumerate(v_1):
            if i>index: flow = flow+1
            else: continue

        #The number of vehicles on the changes in the second road
        for index,i in enumerate(v_2):
            if i>index: flow = flow+1
            else: continue
        
        flow_arr.append(flow)
     
    flow_number = sum(flow_arr)
    flow_number_mean = np.mean(flow_arr)
    car_string = (np.mean(k1_arr)+np.mean(k2_arr))
    car_number = car_mat.ravel().sum()
    car_string_rate = car_string/car_number
    
    return [car_string,car_number,car_string_rate,flow_number,flow_number_mean]



    




