
import numpy as np
import pandas as pd
import re
import os
import imp
import numpy as np
import copy
import warnings
warnings.filterwarnings('ignore') 
from multiprocessing import Pool


class PrematureDeath:
    
    def __init__(self, data_protect_population, data_heterogeneity, data_pollution, data_info, data_IER):
        self.df_pp = data_protect_population.copy()
        self.df_h = data_heterogeneity.copy()
        self.df_pm = data_pollution.copy()
        self.df_info = data_info.copy()
        self.df_IER = data_IER.copy()
        self.pre_data()
        
    def pre_data(self):
        self.df_main = self.df_pm.merge(self.df_pp.drop(columns='province'), on='city')
        self.df_main['pred_outdoor_time'] = self.df_main['pred_outdoor_time'] / 24
        self.df_main['pred_air_clean'] = (self.df_main['pred_work_airclean'] + self.df_main['pred_home_airclean']) / 2
        self.df_main = self.df_main.merge(self.df_info, on='city', how='left')
        self.df_main['attention_decay'] = self.df_main['hot_value'] / self.df_main['population']
        bj_index = copy.deepcopy(self.df_main.query("city=='北京'")['attention_decay'].values[0])
        self.df_main['attention_decay'] = self.df_main['attention_decay'] / bj_index
        self.df_h['air_clean'] = (self.df_h['work_airclean'] + self.df_h['home_airclean']) / 2
        self.df_h['outdoor_time'] = self.df_h['outdoor_time'] / 24
        self.IER_params = {}
        self.df_IER['cause'] = self.df_IER['cause'].replace({'STROKE': 'Stroke'})
        for i in ['IHD', 'COPD', 'Stroke', 'LC']:
            self.IER_params[i] = self.df_IER.query("age in ['AllAge', 'All Age']").query(f"cause=='{i}'")[['alpha','beta','delta','zcf']].to_dict('records')
        # print(self.df_main.query("city=='北京' and date==20200101").to_dict('records'))
    def RR_model(self, pm_value, alpha, gamma, theta, c_0):
        if pm_value > c_0:
            RR_value = 1 + alpha * (1 - np.exp(-gamma * (pm_value - c_0)**theta))
        else:
            RR_value = 1
        return RR_value
    
    def AC_model(self, RR_value, population, incidence_rate):
        #print(RR_value, population, incidence_rate)
        AC_value = (RR_value - 1) / RR_value * population * incidence_rate
        #print(AC_value)
        return AC_value
    
    def single_params_model(self, pm_value, population, disease_type):
        
        if disease_type == 'IHD':
            p_random = self.IER_params[disease_type][np.random.randint(0, 1000)]
            RR_value = self.RR_model(pm_value, p_random['alpha'], p_random['beta'], p_random['delta'], p_random['zcf'])
            AC_value = self.AC_model(RR_value, population, 0.000707)
        elif disease_type == 'Stroke':
            p_random = self.IER_params[disease_type][np.random.randint(0, 1000)]
            RR_value = self.RR_model(pm_value, p_random['alpha'], p_random['beta'], p_random['delta'], p_random['zcf'])
            AC_value = self.AC_model(RR_value, population, 0.00129)
        elif disease_type == 'COPD':
            p_random = self.IER_params[disease_type][np.random.randint(0, 1000)]
            RR_value = self.RR_model(pm_value, p_random['alpha'], p_random['beta'], p_random['delta'], p_random['zcf'])
            AC_value = self.AC_model(RR_value, population, 0.000696)
        elif disease_type == 'LC':
            p_random = self.IER_params[disease_type][np.random.randint(0, 1000)]
            RR_value = self.RR_model(pm_value, p_random['alpha'], p_random['beta'], p_random['delta'], p_random['zcf'])
            AC_value = self.AC_model(RR_value, population, 0.000383)
        else:
            raise ValueError("No such disease !")

        return AC_value
    
    def random(self, mean, std, num=0):
        num += 1
        value = np.random.normal(mean, std)
        if value > 0 and value < 1:
            return value
        elif value > 0 and num > 10:
            return mean
        return self.random(mean, std, num)
    
    def random2(self, mean, std):
        value = np.random.normal(mean, std)
        if value > 0:
            return value
        return self.random(mean, std)
            
    
    def gen_decay_pm(self, x, scene, std=0.2):
        # 室内衰减
        if x['pred_mask_ratio'] >= 1:
            x['pred_mask_ratio'] == 0.97
        #"""
        if x['pred_air_clean'] >= 1:
            x['pred_air_clean'] == 0.95
        if x['pred_outdoor_time'] >= 1:
            x['pred_outdoor_time'] == 0.95
        #"""
        if scene == 0:
            # 不考虑任何防护措施, 室内室外
            decay_indoor = 0
            decay_clean = 1
            decay_mask = 1
            mask_ratio = 0
            outdoor_ratio = 1
            clean_ratio = 0
        elif scene == 1:
            # 不考虑任何防护措施
            decay_indoor = self.random(x['inf_factor'], std)
            decay_clean = 1
            decay_mask = 1
            mask_ratio = 0
            outdoor_ratio = self.random(x['pred_outdoor_time'], std)
            clean_ratio = 0
        elif scene == 2:
            # 考虑所有防护措施
            if x['pollution_grade'] >= 2:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = self.random(0.2, std)
                decay_mask = self.random(0.3, std)
                mask_ratio = self.random(x['pred_mask_ratio'], std)
                outdoor_ratio = self.random(x['pred_outdoor_time'] * (1 - x['pred_cancel_outdoor']), std)
                clean_ratio = self.random(x['pred_air_clean'], std)
            else:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = 0
                decay_mask = 1
                mask_ratio = 0
                outdoor_ratio = self.random(x['pred_outdoor_time'], std)
                clean_ratio = 0
        elif scene == 3:
            # 实际估算
            if x['pollution_grade'] >= 2:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = self.random(0.2, std)
                decay_mask = self.random(0.3, std)
                mask_ratio = self.random(x['pred_mask_ratio'] * x['pred_haze_attention'], std)
                outdoor_ratio = self.random(x['pred_outdoor_time'] * (1 - x['pred_haze_attention'] * x['pred_cancel_outdoor']), std)
                clean_ratio = self.random(x['pred_air_clean'] * x['pred_haze_attention'], std)
            else:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = 0
                decay_mask = 1
                mask_ratio = 0
                outdoor_ratio = self.random(x['pred_outdoor_time'], std)
                clean_ratio = 0
        elif scene == 4:
            # 按美国空气质量标准
            new_pollution_standard = [[0, 12], [13, 35], [36, 55], [56, 150], [151, 250], [251, np.inf]]
            for i, s in enumerate(new_pollution_standard):
                if s[0] < x['PM25'] < s[1]:
                    if i > x['pollution_grade']:
                        x['pollution_grade'] = i
            
            if x['pollution_grade'] >= 2:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = self.random(0.2, std)
                decay_mask = self.random(0.3, std)
                mask_ratio = self.random(x['pred_mask_ratio'] * x['pred_haze_attention'], std)
                outdoor_ratio = self.random(x['pred_outdoor_time'] * (1 - x['pred_haze_attention'] * x['pred_cancel_outdoor']), std)
                clean_ratio = self.random(x['pred_air_clean'] * x['pred_haze_attention'], std)
            else:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = 0
                decay_mask = 1
                mask_ratio = 0
                outdoor_ratio = self.random(x['pred_outdoor_time'], std)
                clean_ratio = 0
        elif scene == 5:
            beijing = {
                'pred_outdoor_time': 0.2815285055272793, 'pred_mask_ratio': 0.979189152893585, 'pred_cancel_outdoor': 0.6825034134499437, 
                'pred_haze_attention': 0.6016977513190402,'pred_air_clean': 0.8484315423550237
            }
            if x['pollution_grade'] >= 2:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = self.random(0.2, std)
                decay_mask = self.random(0.3, std)
                mask_ratio = self.random(beijing['pred_mask_ratio'] * beijing['pred_haze_attention'], std)
                outdoor_ratio = self.random(beijing['pred_outdoor_time'] * (1 - beijing['pred_haze_attention'] * beijing['pred_cancel_outdoor']), std)
                clean_ratio = self.random(beijing['pred_air_clean'] * beijing['pred_haze_attention'], std)
            else:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = 0
                decay_mask = 1
                mask_ratio = 0
                outdoor_ratio = self.random(beijing['pred_outdoor_time'], std)
                clean_ratio = 0
        else:
            raise ValueError("scene error !")
        x['PM25'] = self.random2(x['PM25'], x['std_d'])
        pm_weight = x['PM25'] * (1 + (decay_mask - 1) * mask_ratio) * outdoor_ratio + x['PM25'] * decay_indoor * (1 - outdoor_ratio) * (1 + clean_ratio * (decay_clean - 1))
        return pm_weight 
    
    def heterogeneity_decay_pm(self, x, params, scene, type_, std=0.2):
        if x['pred_air_clean'] >= 1:
            x['pred_air_clean'] == 0.99
        if scene == 0:
            # 不考虑任何防护措施，室内室外
            decay_indoor = self.random(x['inf_factor'], std)
            decay_clean = 1
            decay_mask = 1
            mask_ratio = 0
            outdoor_ratio = 1
            clean_ratio = 0
        elif scene == 1:
            decay_indoor = self.random(x['inf_factor'], std)
            decay_clean = 1
            decay_mask = 1
            mask_ratio = 0
            outdoor_ratio = self.random(params['outdoor_time'], std)
            clean_ratio = 0
        elif scene == 2:
            if x['pollution_grade'] >= 2:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = self.random(0.2, std)
                decay_mask = self.random(0.3, std)
                mask_ratio = self.random(params['mask_ratio'], std)
                outdoor_ratio = self.random(params['outdoor_time'] * (1 - params['cancel_outdoor']), std)
                if type_ == 'is_city':
                    clean_ratio = self.random(params['air_clean'], std)
                else:
                    clean_ratio = self.random(x['pred_air_clean'], std)
            else:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = 0
                decay_mask = 1
                mask_ratio = 0
                outdoor_ratio = self.random(params['outdoor_time'], std)
                clean_ratio = 0
        elif scene == 3:
            if x['pollution_grade'] >= 2:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = self.random(0.2, std)
                decay_mask = self.random(0.3, std)
                mask_ratio = self.random(params['mask_ratio'] * params['haze_attention'], std)
                outdoor_ratio = self.random(params['outdoor_time'] * (1 - params['haze_attention'] * params['cancel_outdoor']), std)
                if type_ == 'is_city':
                    clean_ratio = self.random(params['air_clean'] * params['haze_attention'], std)
                else:
                    clean_ratio = self.random(x['pred_air_clean'] * params['haze_attention'], std) 
            else:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = 0
                decay_mask = 1
                mask_ratio = 0
                outdoor_ratio = self.random(params['outdoor_time'], std)
                clean_ratio = 0
        elif scene == 4:
             # 按美国空气质量标准
            new_pollution_standard = [[0, 12], [13, 35], [36, 55], [56, 150], [151, 250], [251, np.inf]]
            for i, s in enumerate(new_pollution_standard):
                if s[0] < x['PM25'] < s[1]:
                    if i > x['pollution_grade']:
                        x['pollution_grade'] = i
            if x['pollution_grade'] >= 2:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = self.random(0.2, std)
                decay_mask = self.random(0.3, std)
                mask_ratio = self.random(params['mask_ratio'] * params['haze_attention'], std)
                outdoor_ratio = self.random(params['outdoor_time'] * (1 - params['haze_attention'] * params['cancel_outdoor']), std)
                if type_ == 'is_city':
                    clean_ratio = self.random(params['air_clean'] * params['haze_attention'], std)
                else:
                    clean_ratio = self.random(x['pred_air_clean'] * params['haze_attention'], std) 
            else:
                decay_indoor = self.random(x['inf_factor'], std)
                decay_clean = 0
                decay_mask = 1
                mask_ratio = 0
                outdoor_ratio = self.random(params['outdoor_time'], std)
                clean_ratio = 0
        else:
            raise ValueError("scene error !")
        x['PM25'] = self.random2(x['PM25'], x['std_d'])
        pm_weight = x['PM25'] * (1 + (decay_mask - 1) * mask_ratio) * outdoor_ratio + x['PM25'] * decay_indoor * (1 - outdoor_ratio) * (1 + clean_ratio * (decay_clean - 1))
        return pm_weight
    
    def simulate(self, task_name, scene=None, type_=None, class_=None, disease_type=None,sim_times=1000):
        """
        city|scene_1_IHD_mean|scene_1_Stroke_mean|scene_1_COPD_mean|scene_1_LC_mean|scene_1_IHD_low|scene_1_Stroke_low|scene_1_COPD_low|scene_1_LC_low|scene_1_IHD_high|scene_1_Stroke_high|scene_1_COPD_high|scene_1_LC_high|
        """
        df_result = pd.DataFrame({})
        if task_name == 'scene':
            cities = list(set(self.df_main['city']))
            for n, city in  enumerate(cities):
                values = {disease_type: [], f'PM25_{disease_type}': []}
                df_target_city = self.df_main.query(f"city=='{city}'").copy()
                for t in range(sim_times):
                    df_target_city['pm_temp'] = df_target_city.apply(lambda x: self.gen_decay_pm(x, scene=scene), axis=1)
                    values[disease_type].append(self.single_params_model(df_target_city['pm_temp'].mean(), df_target_city['population'].values[-1], disease_type=disease_type))
                    values[f'PM25_{disease_type}'].append(df_target_city['pm_temp'].mean())
                ac_values = {'city': city}
                for i in values.keys():
                    std = np.array(values[i]).std(ddof=1)
                    mean = np.array(values[i]).mean()
                    ci_down = mean - 1.96 * std
                    ci_up = mean + 1.96 * std
                    ac_values['scene' + '_' + str(scene) + '_' + i + '_' + 'mean'] = [mean]
                    ac_values['scene' + '_' + str(scene) + '_' + i + '_' + 'low'] = [ci_down]
                    ac_values['scene' + '_' + str(scene) + '_' + i + '_' + 'high'] = [ci_up]
                df_ac_values = pd.DataFrame(ac_values)
                df_result = pd.concat([df_result, df_ac_values])
            print(scene, disease_type)

        elif task_name == 'heterogeneity':
            cities = list(set(self.df_main['city']))
            params = self.df_h.query(f"type_=='{type_}' and class_=='{class_}'").to_dict('records')[0]
            for n, city in enumerate(cities):
                values = {disease_type: [], f'PM25_{disease_type}': []}
                df_target_city = self.df_main.query(f"city=='{city}'").copy()
                for t in range(sim_times):
                    df_target_city['pm_temp'] = df_target_city.apply(lambda x: self.heterogeneity_decay_pm(x, params, scene, type_), axis=1)
                    c_mean = df_target_city['pm_temp'].mean()
                    values[disease_type].append(self.single_params_model(c_mean, df_target_city[class_].values[-1], disease_type=disease_type))
                    values[f'PM25_{disease_type}'].append(c_mean)
                ac_values = {'city': city}
                for i in values.keys():
                    std = np.array(values[i]).std(ddof=1)
                    mean = np.array(values[i]).mean()
                    ci_down = mean - 1.96 * std
                    ci_up = mean + 1.96 * std
                    ac_values[f'scene_{scene}_' + type_ + '_' + class_ + '_' + i + '_' + 'mean'] = [mean]
                    ac_values[f'scene_{scene}_' + type_ + '_' + class_ + '_' + i + '_' + 'low'] = [ci_down]
                    ac_values[f'scene_{scene}_' + type_ + '_' + class_ + '_' + i + '_' + 'high'] = [ci_up]
                df_ac_values = pd.DataFrame(ac_values)
                df_result = pd.concat([df_result, df_ac_values])
            print(scene, disease_type)
        else:
            raise KeyError("输入任务名称错误，scene或者heterogeneity")
        return df_result
    
    def gen_result(self, task_type, sim_times=1000):
        if task_type == 'scene':
            df_scene_all = None
            for i in range(1, 5):
                print("scene", i)
                df_scene_temp = self.simulate(task_name=task_type, scene=i, sim_times=sim_times)
                if i == 1:
                    df_scene_all = df_scene_temp
                else:
                    df_scene_all = df_scene_all.merge(df_scene_temp, on='city')
            df_scene_all.to_excel("result/df_all_scene_test.xlsx", index=False)
        elif task_type == 'heterogeneity':
            df_h_all = None
            for pair in zip(self.df_h['type_'], self.df_h['class_']):
                print("pair", pair, scene, disease_type)
                df_h_temp = self.simulate(task_name='heterogeneity', scene=scene, type_=pair[0], class_=pair[1], sim_times=sim_times)
                if df_h_all is None:
                    df_h_all = df_h_temp
                else:
                    df_h_all = df_h_all.merge(df_h_temp, on='city')
            
            df_h_all.to_excel("result/df_all_heterogeneity_test.xlsx", index=False)
        else:
            pass
    
    def task(self, args):
        print(args)
        task_type = args[2]
        if task_type == 'scene':
            scene = args[0]
            disease_type = args[1]
            sim_times = args[3]
            df_scene_temp = self.simulate(task_name=task_type, disease_type=disease_type, scene=scene, sim_times=sim_times)
        elif task_type == 'heterogeneity':
            scene = args[0]
            disease_type = args[1]
            sim_times = args[3]
            pair = args[4]
            df_scene_temp = self.simulate(task_name=task_type, disease_type=disease_type, scene=scene, type_=pair[0], class_=pair[1], sim_times=sim_times)
        else:
            raise ValueError
        return df_scene_temp
        
    
def start_task(task_type, df_protect, df_group, df_pm, df_info, df_IER, sim_times=1000):
    params  = []
    if task_type == 'scene':
        cpu_worker = 20
        for i in range(6):
            for j in ['IHD', 'Stroke', 'COPD', 'LC']:
                params.append((i, j, task_type, sim_times))
    else:
        cpu_worker = 24
        for i in range(5):
            for j in ['IHD', 'Stroke', 'COPD', 'LC']:
                for pair in zip(df_group['type_'], df_group['class_']):
                    params.append((i, j, task_type, sim_times, pair))
    pmd = PrematureDeath(df_protect, df_group, df_pm, df_info, df_IER)
    with Pool(cpu_worker) as p:
        outputs = p.map(pmd.task, params)
    df_all =None
    for i in outputs:
        if df_all is None:
            df_all = i
        else:
            df_all = df_all.merge(i)
    return df_all
"""
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # df_q = pd.read_excel("data/问卷数据整理.xlsx")
    df_protect = pd.read_excel("data/result_预测防护数据.xlsx")
    df_pm = pd.read_csv("data/城市空气质量等级以及PM25数据.csv")
    df_group = pd.read_excel("data/不同类别人群防护异质数据2.xlsx")
    df_info = pd.read_csv("data/all_information_distribute.csv")
    df_info['hot_value'] = df_info['value'] + df_info['b_value']
    df_info = df_info.query("year>=2019").groupby(['province', 'city'])['hot_value'].mean().reset_index()
    df_IER = pd.read_csv("data/IER参数/IHME_CRCurve_parameters.csv")
    df = start_task('scene', df_protect, df_group, df_pm, df_info, df_IER, sim_times=1000)
    df.to_excel("scene_test.xlsx", index=False)
"""
