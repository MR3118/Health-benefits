import numpy as np
import pandas as pd
import re

def gen_regression_data(data_city_panel, data_scene, data_info):
    df_city_death = pd.DataFrame({})
    for scene in range(5):
        for disease_type in ['IHD', 'Stroke', 'COPD', 'LC']:
            df_temp = data_scene.filter(regex=f"scene_{scene}_{disease_type}.*?").copy()
            df_temp['city'] = data_scene['city']
            df_temp['scene'] = f'S{scene}'
            df_temp = df_temp.groupby(['city', 'scene']).sum().reset_index()
            df_temp['disease_type'] = disease_type
            df_temp.rename(columns={f"scene_{scene}_{disease_type}_mean": 'value', f"scene_{scene}_{disease_type}_low": 'low', f"scene_{scene}_{disease_type}_high": 'up'}, inplace=True)
            df_city_death = pd.concat([df_city_death, df_temp])
    df_city_pm = pd.DataFrame({})
    for scene in range(5):
        for disease_type in ['IHD', 'Stroke', 'COPD', 'LC']:
            df_temp = data_scene.filter(regex=f"scene_{scene}_PM25_{disease_type}.*?").copy()
            df_temp['city'] = data_scene['city']
            df_temp['scene'] = f'S{scene}'
            df_temp = df_temp.groupby(['city', 'scene']).sum().reset_index()
            df_temp['disease_type'] = disease_type
            df_temp.rename(columns={f"scene_{scene}_PM25_{disease_type}_mean": 'value', f"scene_{scene}_PM25_{disease_type}_low": 'low', f"scene_{scene}_PM25_{disease_type}_high": 'up'}, inplace=True)
            df_city_pm = pd.concat([df_city_pm, df_temp])
    df_city_pm_g = df_city_pm.groupby(['city', 'scene'])['value'].mean().reset_index().rename(columns={'value': 'PM25'}).query("scene=='S0'").drop(columns='scene')
    df_city_death = df_city_death.groupby(['city', 'scene'])['value'].sum().reset_index()
    df_cd_s1 = df_city_death.query("scene=='S1'").copy().rename(columns={'value': 'value_1'}).drop(columns='scene')
    df_cd_s3 = df_city_death.query("scene=='S3'").copy().drop(columns='scene')
    df_cd_s3 = df_cd_s3.merge(df_cd_s1, on='city')
    df_cd_s3['death_reduce_value'] = df_cd_s3['value'] - df_cd_s3['value_1']
    df_cd_s3['death_reduce_value'] = df_cd_s3['death_reduce_value'].map(lambda x: 0 if x > 0 else x)
    df_cd_s3['death_reduce_rate'] = df_cd_s3['death_reduce_value'] / (df_cd_s3['value_1'] + 1)
    df_cd_s3.rename(columns={'value': 'death_value_3', 'value_1': 'death_value_1'}, inplace=True)
    
    
    df_info_2020 = data_info.query("year==2019").copy()
    df_info_2020 = df_info_2020.groupby('city')[['value', 'b_value']].sum().reset_index()
    df_info_2020['sum_info_value'] = df_info_2020['value'] + df_info_2020['b_value']
    df_info_2020.rename(columns={'value': 'weibo_value', 'b_value': 'baidu_value'}, inplace=True)
    df_result = pd.merge(df_cd_s3, df_info_2020, on='city').merge(data_city_panel, on='city').merge(df_city_pm_g, on='city')
    
    df_result['log_death_reduce_value'] = np.log(-df_result['death_reduce_value'] + 1)
    df_result['log_GDP_mean'] = np.log(df_result['GDP'] / df_result['population_x'] * 10000 + 1)
    df_result['log_GDP'] = np.log(df_result['GDP'] + 1)
    df_result['sum_info_value_mean'] = df_result['sum_info_value'] / df_result['population_x']
    df_result['log_sum_info_value_mean'] = np.log(df_result['sum_info_value_mean']  + 1)
    df_result['log_sum_info_value'] = np.log(df_result['sum_info_value'] + 1)
    df_result['log_hospital_beds'] = np.log(df_result['hospital_beds'] + 1)
    df_result['log_population_x'] = np.log(df_result['population_x'] + 1)
    df_result['log_green_ground'] = np.log(df_result['green_ground'] + 1)
    df_result['log_gov_general_spend'] = np.log(df_result['gov_general_spend'] + 1)
    df_result['log_industry_company'] = np.log(df_result['industry_company'] + 1)
    df_result['log_oil_supply'] = np.log(df_result['oil_supply'] + 1)
    df_result['log_income'] = np.log(df_result['income'] + 1)
    df_result['log_PM25'] = np.log(df_result['PM25']+1)
    df_result['log_weibo_value'] = np.log(df_result['weibo_value'] / df_result['population_x']  + 1)
    df_result['log_baidu_value'] = np.log(df_result['baidu_value'] / df_result['population_x']  + 1)
    df_result['total_info'] = df_result['sum_info_value'].sum()
    df_result['total_pm'] = df_result['PM25'].sum()
    df_result['info_index'] = (df_result['sum_info_value'] / df_result['total_info']) / (df_result['PM25'] / df_result['total_pm'])
    df_result['death_rate'] = -df_result['death_reduce_value'] / df_result['population_x']
    df_result['log_death_rate'] = np.log(df_result['death_rate'] + 1)
    df_result['log_info_index'] = np.log(df_result['info_index'] + 1)
    df_result['log_death_reduce_rate'] = np.log(-df_result['death_reduce_rate'] + 1)
    return df_result