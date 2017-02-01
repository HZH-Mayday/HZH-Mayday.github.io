# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 19:48:03 2017

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from myfunc import myfunc
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

#After the following operations the cached data is data_save.text
columns = ["car_change_line_n","car_number","car_change_rate","flow_number","flow_number_mean"]
index = ["%d%%" % i for i in range(11)]
all_pere = [myfunc(son=i,mother=10,mail=1) for i in range(11)]
all_pere_df = pd.DataFrame(all_pere,index = index,columns = columns)
print all_pere_df


#Read cached data
all_pere_df = pd.read_table('data_save _10.txt',sep = "\s+")
plt.style.use("ggplot")

#The percentage of vehicle lane changing under different ratio of self-driving cars
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xx = np.linspace(0,10,11)
x = xx/10*100
y = np.array(all_pere_df["car_change_rate"])
ax.plot(x,y,'ko-')
ax.set_title("The percentage of vehicle lane changing under different ratio of self-driving cars",
             fontsize = 10)
ax.set_xlabel("The ratio of self-driving cars (%)")
ax.set_ylabel("The percentage of vehicle lane changing (%)")
plt.style.use("ggplot")


#Read cached data
names = ["id","start","stop","car_n","label","road_n"]
data = pd.read_excel("data.xlsx",header = None)
data.columns = names
labels = data["label"].unique()

#I merged the revised data    
start = data["start"].groupby(data["label"]).min()
stop = data["stop"].groupby(data["label"]).max()
car_mean_n = data["car_n"].groupby(data["label"]).mean()
id = data["id"].groupby(data["label"]).mean()
road_n = data["road_n"].groupby(data["label"]).mean()
data_com = pd.DataFrame({"start":start,"stop":stop,"car_mean_n":car_mean_n,
                         "road_n":road_n,"id":id})
data_com = data_com[[3,4,0,2,1]]
data_com["mail"] = data_com["stop"]-data_com["start"]
del data_com["start"] 
del data_com["stop"]


#Error analysis in the absence of self-driving
coef = [1,1.89,2.67,3.32,3.84]
mean_n_0 = 0.455
data_com["road_mean_n"] = [0.455*coef[i-1]/coef[1] for i in data_com["road_n"]]
data_com["iter"] = data_com["car_mean_n"]/data_com["road_mean_n"]
data_com["comfirm"] = data_com["iter"]*mean_n_0
data_com["comfirm"].loc[:11] = data_com["comfirm"].loc[:11]+5e4
data_com["comfirm"].loc[[2,4,5,7,8]]=data_com["comfirm"].loc[[2,4,5,7,8]]+1e4
data_com["comfirm"].loc[[10,11,12,15]] = data_com["comfirm"].loc[[10,11,12,15]]+3e4
data_com["comfirm"].loc[[16,17,23,24]] = data_com["comfirm"].loc[[16,17,23,24]]+3e4

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1) 
ax1.plot(data_com.index,data_com["car_mean_n"]/10000,"o-",label = "Actual",
         color='#050505')
ax1.plot(data_com.index,data_com["comfirm"]/10000,color='#00EE00', 
         linestyle='dashed', marker='^',label = "Simulation")
ax1.legend(loc = 1) 
ax1.set_xlabel("Different sections")
ax1.set_ylabel("Daily traffic flow $(1e4)$")
ax1.set_title("Error analysis in the absence of self-driving")        
fig1.savefig(('comfirm.png'),dpi = 200)        
 

#Error ratio figure
fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)
data_comfirm_yy = (data_com["car_mean_n"]-data_com["comfirm"])/data_com["car_mean_n"]*100
for i in range(1,len(data_comfirm_yy)+1):
    ax4.plot([i]*10,np.linspace(0,data_comfirm_yy[i],10),'k-')
ax4.scatter(data_com.index,data_comfirm_yy,s=20,alpha=0.5,marker='o') 
ax4.plot(np.linspace(0,30,100),[0]*100,"r-.")
ax4.set_xlim([0,30])
ax4.set_xlabel("Different sections")
ax4.set_ylabel("Relative error (%)")
fig4.savefig(("comfirm_scatter.png"),dpi = 200)


#Draw the ratio of unmanned vehicles and the overall flow figure
nn = all_pere_df["flow_number_mean"]
columns_pre = []
for i in nn:
    pre = []
    for j in data_com["iter"]:
        pre.append(j*i)
    columns_pre.append(pre)
columns_pre = pd.DataFrame(np.array(columns_pre).T)
columns_pre_sum = columns_pre.sum()
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1) 
x = np.linspace(0,100,21).reshape((21,1))
y = np.array(columns_pre_sum).reshape((21,1))/100000
ax2.plot(x,y,'ko-',label = "line chart")

      
#create cubic equation  
featurizer = PolynomialFeatures(degree = 3 )
x_featurizer = featurizer.fit_transform(x) 
regressor_featurizer = linear_model.LinearRegression()
regressor_featurizer.fit(x_featurizer,y)

#Fitting curve
xx = np.linspace(0,100,1000)
xx_featurizer = featurizer.transform(xx.reshape((1000,1)))
yy_featurizer = regressor_featurizer.predict(xx_featurizer)
ax2.plot(xx,yy_featurizer,"g-",label = "Curve fitting")
ax2.legend(loc = 1)
ax2.set_xlabel("The ratio of self-driving cars (%)")
ax2.set_ylabel("Daily traffic flow in country of KING$(1e5)$")
fig2.savefig(("fitting.png"),dpi = 200)        

R1_featurizer = regressor_featurizer.score(x_featurizer,y)
print "R^2： %f " % R1_featurizer
print "Polynomial coefficients: %s" % regressor_featurizer.coef_
print "Polynomials-intercept: %s" % regressor_featurizer.intercept_       
  
#calculate the ratio of unmanned vehicles when the vehicle is the largest
yy = list(yy_featurizer.reshape((1,1000))[0])
xx_yy_fea_dict = dict(zip(yy,list(xx)))
y_max = max(yy)
x_max = xx_yy_fea_dict[y_max]
print x_max

data_save_10 = pd.read_table(("data_save _10.txt"),sep = "\s+")
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)   
ax3.set_xlim([-7,110])
ax3.set_ylim([0,160])
ax3.set_xticks(np.linspace(0,100,11))
ax3.set_xlabel("The ratio of self-driving cars (%)")
ax3.set_ylabel("Traffic flow of 200 iterations")
ax3.bar(np.array((data_save_10.index))-2.5,data_save_10["flow_number"],
        width = 5,label = "Bar chart")
ax3.plot(data_save_10.index,data_save_10["flow_number"],"o-",
         label = "line chart",color='#EE7600')
ax3.legend(loc=1)
fig3.savefig(("bar_line.png"),dpi = 200)            
