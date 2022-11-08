import numpy as np
import pandas as pd
import re
import os
import glob
import matplotlib.pyplot as plt
from pyecharts.charts import Bar, Map, Line
import imp
import code_.aqi as aqi
import code_.utils as utils
import code_.yearbook as yearbook
import numpy as np
import plotnine as pn
from code_ import id2url
import code_.weibo as weibo
import matplotlib.image as img
import sqlite3
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from scipy.linalg import fractional_matrix_power
imp.reload(aqi)
imp.reload(yearbook)
imp.reload(utils)
imp.reload(weibo)
pd.set_option("display.max_columns", None)



class InferWork:
    
    def __init__(self, df_q, df_province_stat, df_province_7, df_city_stat, df_city_7, df_city_income, df_pm, df_hot):
        self.df_q = df_q.copy()
        self.df_province_stat = df_province_stat.copy()
        self.df_province_7 = df_province_7.copy()
        self.df_city_stat = df_city_stat.copy()
        self.df_city_7 = df_city_7.copy()
        self.df_pm = df_pm.copy()
        self.df_hot = df_hot.copy()
        self.pre_data()
        self.gen_source_data()
        
    def pre_data(self):
        self.df_q['haze_attention'] = self.df_q['haze_attention'].map(lambda x: 0 if x <= 3 else 1)
        same_field = ['PM2.5', 'hot_value']
        province_stat_use = ['province', '地区生产总值(亿元)', '城市绿地面积(万公顷)', 'hospital_beds',
                             '医疗卫生机构床位数(万张)', '地方财政一般预算支出(亿元)',
                             '城市液化石油气供气总量(万吨)', '规模以上工业企业单位数(个)', '第一产业增加值(亿元)', '第二产业增加值(亿元)', 
                             '第三产业增加值(亿元)', '城镇居民人均可支配收入(元)', '农村居民人均可支配收入(元)', 'income', 'edu_year'
                            ]
        city_stat_use = ['city', 'GDP', '绿地面积 (公顷)', 
                '医院床位数 (张)', '地方一般公共预算支出(万元)',
                '液化石油气供气总量(吨)', '工业企业数', '第一产业占地区生产总值的比重（%）','第二产业占地区生产总值的比重（%）',
                '第三产业占地区生产总值的比重（%）', 'city_income_total'
               ]
        # 省份数据
        self.df_province_stat = self.df_province_stat[province_stat_use]
        self.df_province = self.df_province_stat.merge(self.df_province_7, on='province')
        for i in ['第一产业增加值(亿元)', '第二产业增加值(亿元)', '第三产业增加值(亿元)']:
            self.df_province[i] = self.df_province[i] / self.df_province['地区生产总值(亿元)'] * 100
        for i in ['0-14岁', '15-64岁', '65岁及以上']:
            self.df_province[i+'_ratio'] = self.df_province[i] / self.df_province['人口数'] * 100
        self.df_province['female_rate'] = self.df_province['女'] / (self.df_province['女'] + self.df_province['男']) * 100
        self.df_province.rename(columns={
            '第一产业增加值(亿元)': 'first_industry_ratio', 
            '第二产业增加值(亿元)': 'second_industry_ratio', 
            '第三产业增加值(亿元)': 'third_industry_ratio',
            '地区生产总值(亿元)': 'GDP',
            '地方财政一般预算支出(亿元)': 'gov_general_spend',
            '城市绿地面积(万公顷)': 'green_ground',
            '城市液化石油气供气总量(万吨)': 'oil_supply',
            '规模以上工业企业单位数(个)': 'industry_company',
            '城镇化率': 'town_rate',
            '人口数': 'population',
            '0-14岁_ratio': 'age_0_14_ratio',
            '15-64岁_ratio': 'age_15_64_ratio',
            '65岁及以上_ratio': 'age_65_ratio',
        }, inplace=True)
        self.df_province.drop(columns=['医疗卫生机构床位数(万张)', '城镇居民人均可支配收入(元)','农村居民人均可支配收入(元)', 
                                       '0-14岁','15-64岁','65岁及以上', '大专及以上','高中','初中','小学','男','女','year'], inplace=True)
        
        # 城市数据
        self.df_city_stat = self.df_city_stat[city_stat_use]
        self.df_city = self.df_city_stat.merge(self.df_city_7, on='city')
        self.df_city['green_ground'] = self.df_city['绿地面积 (公顷)'] / 10000
        self.df_city['gov_general_spend'] = self.df_city['地方一般公共预算支出(万元)'] / 10000
        self.df_city['oil_supply'] = self.df_city['液化石油气供气总量(吨)'] / 10000
        self.df_city['age_15_64'] = self.df_city['age_15_59'] + self.df_city['age_60'] - self.df_city['age_65']
        for i in ['age_0_14', 'age_15_64', 'age_65']:
            self.df_city[i+'_ratio'] = self.df_city[i] / self.df_city['population'] * 100
        self.df_city['town_rate'] = self.df_city['城镇人口'] / (self.df_city['城镇人口'] + self.df_city['乡村人口']) * 100
        self.df_city['female_rate'] = self.df_city['female'] / (self.df_city['female'] + self.df_city['male']) * 100
        self.df_city['population'] = self.df_city['population'] / 10000
        self.df_city['hospital_beds'] = self.df_city['医院床位数 (张)'] / 10000
        self.df_city.rename(columns={
            '第一产业占地区生产总值的比重（%）': 'first_industry_ratio', 
            '第二产业占地区生产总值的比重（%）': 'second_industry_ratio', 
            '第三产业占地区生产总值的比重（%）': 'third_industry_ratio',
            '工业企业数': 'industry_company',
            'city_income_total': 'income',
        }, inplace=True)
        self.df_city.drop(columns=['液化石油气供气总量(吨)', '地方一般公共预算支出(万元)', '绿地面积 (公顷)', 
                                   'age_0_14', 'age_15_59', 'age_60', 'age_65', 'age_15_64', '大学', '高中', '初中', '小学', 'male','female','城镇人口','乡村人口', 'province'], inplace=True)
        self.df_pm['city'] = self.df_pm['county'].map(lambda x: cr.city_of_county(x))
        self.df_pm['province'] = self.df_pm['city'].map(lambda x: cr.province_of_city(x))
        self.df_province_pm = self.df_pm.groupby(['province'])['county_pm'].mean().reset_index().rename(columns={'county_pm': 'pm25'})
        self.df_city_pm = self.df_pm.groupby(['city'])['county_pm'].mean().reset_index().rename(columns={'county_pm': 'pm25'})
        self.df_province = self.df_province.merge(self.df_province_pm)
        self.df_city = self.df_city.merge(self.df_city_pm).drop_duplicates(subset='city')
        
        province_cols = list(self.df_province.columns)
        province_cols.remove('province')
        city_cols = ['city'] + province_cols
        self.df_city = self.df_city[city_cols]
        self.features = province_cols
        
    def gen_source_data(self):
        self.Y = ['outdoor_time','mask_ratio', 'cancel_outdoor','work_airclean', 'home_airclean', 'haze_attention']
        self.df_q_g = self.df_q.groupby(['province'])[['outdoor_time','mask_ratio', 'cancel_outdoor','work_airclean', 'home_airclean', 'haze_attention']].mean().reset_index()
        self.valid_province = self.df_q.groupby('province')['city'].count().reset_index().query("city>=20")
        self.invalid_province = self.df_q.groupby('province')['city'].count().reset_index().query("city<20")
        self.source = self.df_province.merge(self.df_q_g)
        self.source_x = self.source[self.features]
        self.source_y = self.source[self.Y]
        self.target = self.df_city
        self.target_x = self.target[self.features]
        #print(self.target_province)
        
    def featrue_transform(self):
        s_min = self.source_x.min()
        s_max = self.source_x.max()
        self.source_x = (self.source_x - s_min) / (s_max - s_min)
        t_min = self.target_x.min()
        t_max = self.target_x.max()
        self.target_x = (self.target_x - t_min) / (t_max - t_min)
        corr_source_x = np.cov(self.source_x.values.T) + np.eye(self.source_x.shape[1])
        corr_target_x = np.cov(self.target_x.values.T) + np.eye(self.target_x.shape[1])
        a_coral = np.dot(fractional_matrix_power(corr_source_x, -0.5), fractional_matrix_power(corr_target_x, 0.5))
        source_x_new = np.real(np.dot(self.source_x, a_coral))
        df_source_new = pd.DataFrame(source_x_new, columns=self.features)
        self.source_new = self.source.copy()
        for i in df_source_new.columns:
            self.source_new[i] = df_source_new[i]
        return self.source_new
    
    def model(self, y_name):
        source_new = self.featrue_transform()
        Xs_new = source_new.merge(self.valid_province)[self.features]
        Ys = source_new.merge(self.valid_province)[y_name]
        best_model = {'y_name': y_name, 'best_score': np.inf, 'model_name': None, 'best_params': None, 'best_model': None}
        for model_type in ['rf']:
            if model_type == 'rf':
                m = RandomForestRegressor()
                reg = GridSearchCV(m, param_grid={'n_estimators': list(range(4, 20, 2)), 'max_depth': [2, 3, 4, 5]}, cv=5, scoring='neg_mean_squared_error')
            elif model_type == 'lasso':
                m = Lasso()
                reg = GridSearchCV(m, param_grid={'alpha': np.linspace(0.0002, 0.1, 500)}, cv=5, scoring='neg_mean_squared_error')
            elif model_type == 'ridge':
                m = Ridge()
                reg = GridSearchCV(m, param_grid={'alpha': np.linspace(1, 10, 100)}, cv=5, scoring='neg_mean_squared_error')
            elif model_type == 'svr':
                m = SVR()
                reg = GridSearchCV(m, param_grid={'C': np.linspace(0.001, 10, 50), 'kernel': ['rbf']}, cv=5, scoring='neg_mean_squared_error')
            else:
                raise KeyError("模型类别错误")
            reg.fit(Xs_new, Ys)
            best_score = abs(reg.best_score_)
            if best_score < best_model['best_score']:
                best_model['best_score'] = best_score
                best_model['model_name'] = model_type
                best_model['best_params'] = reg.best_params_
                best_model['best_model'] = reg.best_estimator_
        print(best_model)
        print(best_model['best_model'].score(Xs_new, Ys), Xs_new.shape)
        target_y = best_model['best_model'].predict(self.target_x)
        df_target = pd.DataFrame({'city': list(self.target['city']), 'pred'+ '_' + y_name: target_y})
        return df_target