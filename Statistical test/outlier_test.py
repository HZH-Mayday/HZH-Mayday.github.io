"""
@author: huzhehui
@function: outlier test
@baidu_wenku: https://wenku.baidu.com/view/ee875edbd15abe23482f4d0f.html
"""

# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson

class outlier_test():
    '''
    对输入的数据计算对应的指标, 并检验是否存在异常值.
    1. 先检验模型的残差是否有自相关性, 若有, 则修正; 若没有, 不做任何处理
    2. 计算对应方程的相关指标
    3. 计算得到所有异常点
    '''
    def __init__(self, X, y):
        '''
        :param X: array-like, GDP
        :param y: array-like, 居民人均收入
        '''
        self.X = X
        self.y = y
        self.para_dict = {}     # 参数字典属性


    def dw_test_correction(self):
        '''
        检验拟合的回归方程残差是否有自相关性
        '''
        results = self.fit_LinearRegression(self.X, self.y)     # 拟合方程
        self.coef_p_rsqr_white_values(results)          # 填充参数字典
        dw = self.dw_test(self.X, self.y)          # 计算dw统计量

        # 检验dw值是否通过检验
        if dw > 2.6 or dw < 1.4:
            print(" DW值 = {}, 所以要进行模型修正 ".format(dw))
            self.para_dict = {}

            # 对X, y进行差分转化
            n = len(self.y)
            p = (np.power(n, 2) * (1 - dw/2) + 4) / (np.power(n, 2) - 4)
            y_new = np.array([self.y[i+1]-self.y[i]*p  for i in range(n-1)])
            X_new = np.array([self.X[i+1]-self.X[i]*p  for i in range(n-1)])

            # 拟合修正方程, 填充参数字典
            corr_results = self.fit_LinearRegression(X_new, y_new)
            self.coef_p_rsqr_white_values(corr_results)
            # 修改回归方程系数
            self.dw_test(X_new, y_new)          # 添加新的DW值
            self.para_dict["const_coef"] /= (1 - p)

            # 检验异常值
            self.outlier = self.outlier_test(corr_results)
        else:
            self.outlier = self.outlier_test(results)


    def fit_LinearRegression(self, x, y):
        '''
        进行线性回归方程的拟合
        :param x: array-like (1,)
        :param y: array-like (1,)
        :return: OLS的拟合结果，方便后续调用
        '''
        x_extant = sm.add_constant(x)
        results = sm.OLS(y, x_extant).fit()
        return results


    def coef_p_rsqr_white_values(self, results):
        '''
        对参数字典进行数据填充
        '''
        white = het_white(results.resid, exog=results.model.exog)[1]
        self.para_dict["f_p"] = results.f_pvalue
        self.para_dict["const_coef"] = results.params[0]
        self.para_dict["x1_coef"] = results.params[1]
        self.para_dict["const_p"] = results.pvalues[0]
        self.para_dict["x1_p"] = results.pvalues[1]
        self.para_dict["r_squared"] = results.rsquared_adj
        self.para_dict["white_p"] = white


    def dw_test(self, x, y):
        '''
        计算dw统计量, 并存入参数字典.
        :param X: array-like, GDP
        :param y: array-like, 居民人均收入
        :return: float, dw_value
        '''
        x1 ,const = self.para_dict["x1_coef"], self.para_dict["const_coef"]
        error = y - (x * x1 + const)
        dw = durbin_watson(error)
        self.para_dict["dw_value"] = dw
        return dw


    def outlier_test(self, results):
        '''
        离群值检验
        :param results: OLS.fit()
        :return: (DataFrame). 包含所有异常值统计量，以及是否为异常值的列. 0(非), 1(是)
        '''
        # 计算异常值检验的统计量
        outliers = results.get_influence()
        leverage = outliers.hat_matrix_diag         # 杠杆值
        resid_stu = outliers.resid_studentized_external         # 学生化残差
        cook = outliers.cooks_distance[0]           # cook距离
        w_k = outliers.dffits[0]           # w-k统计量
        outlier_df = pd.DataFrame({"ti":resid_stu, "cook":cook, "hi":leverage, "wk":w_k})
        outlier_df = outlier_df.applymap(lambda x: round(x, 2))         # 结果保留两位小数

        # 检验是否是异常值, n为数据量, k为解释变量个数, 这里k=1.
        n = len(self.y)
        outlier_df["ti_is_outlier"] = [1 if np.abs(i)>1.729 else 0 for i in outlier_df["ti"]]           # t(n-k-1)=t(19)
        outlier_df["cook_is_outlier"] = [1 if i > 4/n else 0 for i in outlier_df["cook"]]           # 4/n
        outlier_df["hi_is_outlier"] = [1 if i > (2/n) else 0 for i in outlier_df["hi"]]         # 2*k/n
        outlier_df["wk_is_outlier"] = [1 if i > 2*np.sqrt(2/n) else 0 for i in outlier_df["wk"]]            # 2*sqrt((k+1)/n)
        return outlier_df


    def output(self):
        '''
        输出参数 DataFrame
        '''
        self.dw_test_correction()
        df_output = pd.Series(self.para_dict).apply(lambda x: round(x, 2))
        return df_output


if __name__ == "__main__":
    x = np.arange(100)
    y = x*10 + 12 + np.random.randn(100)
    new_test = outlier_test(x, y)
    print(new_test.output())
    print(new_test.para_dict)
