"""
@author: huzhehui
@function: benford law( 本福特定律)
@wiki_from: https://en.wikipedia.org/wiki/Benford%27s_law
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from math import log10

class benford(object):
    
    # 输入数据，以及决定第几位数据
    # data的数据类型为list,number_n数据类型为int
    def __init__(self,data,number_n):    
        self.data = data
        self.number_n = number_n
        self.benford_pro()
        self.actual = self.actual_pro()       # actual--计算出的实际数字概率
        # self.execute_main()                 # 执行全部的检验方式
           
    # 计算对应位数的benford概率
    def benford_pro(self): 
        # 第一位的benford概率
        if self.number_n == 1:
            self.benford = [log10(1+1.0/i) for i in range(1,10)]
         
        # 第二位的benford概率
        elif self.number_n == 2:
            self.benford = []
            for d2 in range(0,10):
                sum_n = 0
                for d1 in range(1,10):
                    sum_n+=log10(1+1.0/(d1*10 + d2))
                self.benford.append(sum_n)
        
        # 第三位的benford概率
        elif self.number_n == 3:            
            self.benford = []
            
            for d3 in range(0,10):
                sum_n = 0
                for d1 in range(1,10):
                    for d2 in range(0,10):
                        sum_n += log10(1+1.0/(d3 + 100*d1 + 10*d2))
                self.benford.append(sum_n) 
                
        self.benford = np.array(self.benford)


    # 计算实际情况下，各位数下，每个数字的概率，pro_df是ndarray类型      
    def actual_pro(self):
        act_pro = np.array([int(str(i)[self.number_n-1]) for i in self.data 
        if len(str(int(i))) >= self.number_n])
        
        self.N = len(act_pro)
        act_pro = act_pro.reshape(self.N,1)
        
        #act_pro = np.array([int(str(i)[self.number_n-1]) for i in self.data]).reshape(self.N,1)
        
        pro_df = pd.DataFrame(act_pro,columns=["number"])
        pro_df = pro_df.groupby(["number"]).size()/(self.N*1.0)
          
        # 如果是第二位或者第三位的数字定律，那么0-9
        if self.number_n == 1:
            pro_df = pro_df.reindex(range(1,10)).fillna(0).values
        else:
            pro_df = pro_df.reindex(range(0,10)).fillna(0).values
            
        return pro_df        
        
        
    # 计算本福特值,分别计算三位数的情况
    # 注意这边的N,是剔除不符合数字位数要求后的N
    def chi_square(self): 
        chi_value = np.sum(np.power(self.actual-self.benford,2)/self.benford)*self.N     
        return chi_value
    
    
    # 计算K_S值,首先确定 实际条件下 和 benford下 的分布函数
    def k_s(self):
        f_actual = np.cumsum(self.actual)
        f_benford = np.cumsum(self.benford)
        VN = np.max(f_actual-f_benford)+np.max(f_benford-f_actual)      # 未经过修正后的K_S值
        K_S_value = VN*(self.N**0.5 + 0.155 + 0.24 * self.N**(-0.5))    # 修正后的K_S值   
        return K_S_value
     
     
    # 距离检验，返回两个值，m和d
    def distance(self):
        m = np.max(np.abs(self.benford-self.actual))
        d = np.sum(np.power(self.benford-self.actual,2))**0.5
        return m,d

    
    # 相关系数检验,求平均
    def pearson(self):
        actual_mean = np.mean(self.actual)
        benford_mean = np.mean(self.benford)
        pearson_cov = np.sum((self.actual-actual_mean)*(self.benford-benford_mean))
        pearson_std = np.sqrt(np.sum(np.square(self.actual-actual_mean))*np.sum(np.square(self.benford-benford_mean)))
        pearson_value = pearson_cov/pearson_std
        return pearson_value
        
    # 综合起来打印出对应的检验值     
    def execute_main(self):
        c = self.chi_square()
        k = self.k_s()
        d = self.distance()
        p = self.pearson()
        result = [round(i, 4) for i in  [p, c, k, d[0], d[1]]]
        return result

def data_make(data):
    '''
    数据处理
    :param data: String
    :return: List
    for example: " 12 123 134 12 " -> [12.0, 123.0, 134.0, 12.0]
    '''
    data_strip = data.strip().replace('\n',',').replace('\t',',').replace('\r',',').replace(' ',',')
    data_list = [float(i) for i in data_strip.split(',') if float(i) != 0]
    return data_list


if __name__ == "__main__":

    data = ''' 12 123 134 12 '''
    benford_c = benford(data_make(data), 1)

    # 按照顺序是  皮尔逊相关系数 -> 卡方值 -> ks值 -> m -> d
    print(benford_c.execute_main())
