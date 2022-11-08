import numpy as np
import pandas as pd
import re
import os
import glob
import matplotlib.pyplot as plt
from pyecharts.charts import Bar, Map, Line, Polar, Grid
from pyecharts import options as opts
import imp
import code_.aqi as aqi
import code_.utils as utils
import code_.yearbook as yearbook
import numpy as np
import plotnine as pn
from code_ import id2url
import code_.weibo as weibo
import matplotlib.image as img
from pyecharts.globals import CurrentConfig, NotebookType
import pypinyin
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
imp.reload(aqi)
imp.reload(yearbook)
imp.reload(utils)
imp.reload(weibo)
pd.set_option("display.max_columns", None)



class PlotResult:
    
    def __init__(self, data_scene, data_heterogenity, data_7_city):
        self.df_scene = data_scene
        self.df_heterogenity = data_heterogenity
        self.df_7_city = data_7_city
        self.remake_scene_data()
        self.remake_heterogenity_data()
        
    def remake_scene_data(self):
        self.df_scene_g = pd.DataFrame({})
        #pm25_cols = list(self.df_scene.filter(regex=f".*?_PM25_.*?").columns)
        for scene in range(6):
            for disease_type in ['IHD', 'Stroke', 'COPD', 'LC']:
                df_s_temp = self.df_scene.filter(regex=f"scene_{scene}_{disease_type}.*?").copy()
                df_s_temp['scene'] = f'S{scene}'
                df_s_temp['city'] = self.df_scene['city']
                df_s_temp = df_s_temp.groupby(['city', 'scene']).sum().reset_index()
                df_s_temp['disease_type'] = disease_type
                df_s_temp.rename(columns={f"scene_{scene}_{disease_type}_mean": 'value', f"scene_{scene}_{disease_type}_low": 'low', f"scene_{scene}_{disease_type}_high": 'up'}, inplace=True)
                #df_s_temp['up'] = df_s_temp['value'] + (df_s_temp['up'] - df_s_temp['value']) * np.power(1000, 0.5) # 模拟计算结果时错误计算了置信区间，这里做反向计算得到正确值
                #df_s_temp['low'] = df_s_temp['value'] - (df_s_temp['value'] - df_s_temp['low']) * np.power(1000, 0.5) # 同上
                self.df_scene_g = pd.concat([self.df_scene_g, df_s_temp])
    
    def remake_heterogenity_data(self):
        self.df_heterogenity_g = pd.DataFrame({})
        for h in [['gender', 'male'], ['gender', 'female'], ['is_city', '乡村人口'], ['is_city', '城镇人口'], ['age', 'age_15_64'], ['age', 'age_0_14'], ['age', 'age_65']]:
            for disease_type in ['IHD', 'Stroke', 'COPD', 'LC']:
                for scene in range(5):
                    df_s_temp = self.df_heterogenity.filter(regex=f"scene_{scene}_{h[0]}_{h[1]}_{disease_type}.*?").copy()
                    cols = list(df_s_temp.columns)
                    #print(df_s_temp)
                    df_s_temp['type_'] = h[0]
                    df_s_temp['class_'] = h[1]
                    df_s_temp['disease_type'] = disease_type
                    df_s_temp['scene'] = scene
                    df_s_temp.rename(columns={f"scene_{scene}_{h[0]}_{h[1]}_{disease_type}_mean": 'value', f"scene_{scene}_{h[0]}_{h[1]}_{disease_type}_low": 'low',
                                              f"scene_{scene}_{h[0]}_{h[1]}_{disease_type}_high": 'up'}, inplace=True)
                    self.df_heterogenity_g = pd.concat([self.df_heterogenity_g, df_s_temp])
                #break
        self.df_heterogenity_g = self.df_heterogenity_g.groupby(['scene', 'type_', 'class_', 'disease_type'])[['value', 'low', 'up']].sum().reset_index()
        self.df_heterogenity_g['class_'].replace({'城镇人口': 'town', '乡村人口': 'rural'}, inplace=True)
        
    def total_death(self, detail=True):
        df_all = self.df_scene_g.groupby(['scene', 'disease_type']).sum().reset_index().copy()
        if detail:
            plot = (
                pn.ggplot(df_all, pn.aes(x='disease_type', y='value', fill='scene'))
                + pn.geom_col(stat='identity', position='dodge', size=4)
                + pn.geom_errorbar(pn.aes(ymin='low', ymax='up'), position = pn.position_dodge(width=0.9), width=0.25)
                + pn.theme(
                    axis_text_x=pn.element_text(size=12), 
                    axis_text_y=pn.element_text(size=12), 
                    axis_title=pn.element_text(size=14),
                    legend_title=pn.element_text(size=12),
                )
                + pn.ylab("")
                + pn.xlab("")
                + pn.labs(fill="Scenes")
            )
            plot.save("论文写作/论文图/图4/disease_death.png", width=5, height=5)
        else:
            df_error = df_all.groupby('scene').sum().reset_index()
            scene_num = df_error.shape[0]
            print(df_error)
            error_low = np.zeros((scene_num, 4))
            error_up = np.zeros((scene_num, 4))
            for i, v in enumerate(zip(list(df_error['up']), list(df_error['low']))):
                error_low[i, 0] = v[1]
                error_up[i, 0] = v[0]
            error_low = list(error_low.reshape(-1))
            error_up = list(error_up.reshape(-1))
            
            plot = (
                pn.ggplot(df_all, pn.aes(x='scene', y='value', fill='disease_type'))
                + pn.geom_col(stat='identity', position='stack', size=4, width=0.25)
                + pn.geom_errorbar(pn.aes(ymin=error_low, ymax=error_up), position = pn.position_dodge(width=0), width=0.25)
                + pn.theme(
                    axis_text_x=pn.element_text(size=12), 
                    axis_text_y=pn.element_text(size=12), 
                    axis_title=pn.element_text(size=14),
                    legend_title=pn.element_text(size=12),
                )
                + pn.ylab("Deaths")
                + pn.xlab(" ")
                + pn.labs(fill="Diseases")
            )
            plot.save("论文写作/论文图/图4/total_death.png", width=5, height=5)
        print(plot)
        df_all = df_all.groupby(['scene', 'disease_type'])[['value', 'up', 'low']].sum().reset_index()
        df_scene = df_all.groupby(['scene'])[['value', 'up', 'low']].sum().reset_index()
        df_scene.columns = ['scene', 'value_total', 'up_total', 'low_total']
        df_all = df_all.merge(df_scene)
        df_all['rate'] = df_all['value'] / df_all['value_total']
        for i in ['S0','S1','S2','S3']:
            df_all_s = df_all.query(f"scene=='{i}'")[['disease_type', 'value_total']].copy().rename(columns={"value_total": f"value_{i}"})
            df_all = df_all.merge(df_all_s)
            #print(df_all)
            df_all[f'reduce_{i}'] = df_all[f'value_{i}'] - df_all[f'value_total'] 
            df_all[f'reduce_rate_{i}'] = df_all[f'reduce_{i}'] / df_all[f'value_{i}']
        return df_all
        
    def heterogenity(self, type_=None, detail=True):
        """废弃"""
        df_g = self.df_heterogenity_g.copy()
        df_g_s0 = df_g.query("scene==0").copy().rename(columns={"value": 's0_value', 'low': 's0_low', 'up': 's0_up'}).drop(columns='scene')
        df_g = df_g.merge(df_g_s0)
        df_g['descend_value'] = df_g['value'] - df_g['s0_value']
        df_g['descend_up'] = df_g['up'] - df_g['s0_up']
        df_g['descend_low'] = df_g['low'] - df_g['s0_low']
        df_g = df_g.query("scene>0")
        #print(df_g)
        if type_:
            df_g_d = df_g.query(f"type_=='{type_}'").copy()
            class_ = list(set(df_g_d['class_']))
            print(class_)
            df_g_d['descend_rate_value'] = (df_g_d['descend_value'] / df_g_d['s0_value'] * 100).map(lambda x: round(np.abs(x), 2))
            df_g_d['descend_rate_up'] = (df_g_d['descend_up'] / df_g_d['s0_up'] * 100).map(lambda x: round(np.abs(x), 2))
            df_g_d['descend_rate_low'] = (df_g_d['descend_low'] / df_g_d['s0_low'] * 100).map(lambda x: round(np.abs(x), 2))
            df_g_1 = df_g_d.query(f"class_=='{class_[0]}'")
            df_g_2 = df_g_d.query(f"class_=='{class_[1]}'")
            # 分疾病类型，以及情景降低百分比图
            plot = (
                pn.ggplot()
                + pn.geom_bar(df_g_1, pn.aes(x='scene', y='descend_rate_value', fill='disease_type'), stat='identity', position=pn.position_dodge(width=0.6), width=0.5)
                + pn.geom_bar(df_g_2, pn.aes(x='scene', y='-descend_rate_value', fill='disease_type'), stat='identity', position=pn.position_dodge(width=0.6), width=0.5)
                + pn.geom_hline(yintercept=0, colour="white", size=2)
                + pn.geom_errorbar(df_g_1, pn.aes(x='scene', ymin='descend_rate_low', ymax='descend_rate_up', fill='disease_type'), position=pn.position_dodge(width=0.6), width=0.25)
                + pn.geom_errorbar(df_g_2, pn.aes(x='scene', ymin='-descend_rate_low', ymax='-descend_rate_up', fill='disease_type'), position=pn.position_dodge(width=0.6), width=0.25)
                + pn.coord_flip()
                + pn.theme_bw()
                + pn.theme(figure_size=(10, 6))
                + pn.ylab(f"<-----  {class_[1]}                             {class_[0]}  ----->")
                + pn.scale_x_continuous(breaks=[1, 2, 3, 4], labels=['S1', 'S2', 'S3', 'S4'])
                + pn.geom_vline(xintercept=[1.5, 2.5, 3.5], colour="black", size=2, linetype='dotted')
                + pn.scale_y_continuous(breaks=list(range(-40, 50, 10)), labels=[f'{i}%' for i in list(range(0, 50, 10))[::-1] + list(range(10, 50, 10))])
            )
            print(plot)
        # 分组 
        else:
            df_g_s = df_g.drop(columns=['scene', 'disease_type']).groupby(['type_', 'class_']).sum().reset_index().copy()
            df_g_s['descend_rate_value'] = (df_g_s['descend_value'] / df_g_s['s0_value'] * 100).map(lambda x: round(np.abs(x), 2))
            df_g_s['descend_rate_up'] = (df_g_s['descend_up'] / df_g_s['s0_up'] * 100).map(lambda x: round(np.abs(x), 2))
            df_g_s['descend_rate_low'] = (df_g_s['descend_low'] / df_g_s['s0_low'] * 100).map(lambda x: round(np.abs(x), 2))
            plot = (
                pn.ggplot(df_g_s, pn.aes(x='type_', y='descend_rate_value', fill='class_'))
                + pn.geom_col(stat='identity', position='dodge', size=4, width=0.25)
                + pn.theme_bw()
                # + pn.geom_errorbar(pn.aes(ymin='descend_rate_low', ymax='descend_rate_up'), position = pn.position_dodge(width=0.25), width=0.1)
            )
            print(plot)
            
    def scene_heterogenity(self, type_, pic_type, div=1):
        df_he = self.df_heterogenity_g.query(f"type_=='{type_}'").copy()
        df_he = df_he.groupby(by=['scene', 'class_'])[['value', 'low', 'up']].sum().reset_index()
        if pic_type == 'absolute':
            plot = (
                pn.ggplot()
                + pn.geom_bar(df_he, pn.aes(x='scene', y='value', fill='class_'), stat='identity', position=pn.position_dodge(width=0.6), width=0.5)
                + pn.geom_errorbar(df_he, pn.aes(x='scene', ymin='low', ymax='up', fill='class_'), position=pn.position_dodge(width=0.6), width=0.25)
                + pn.theme_bw() 
                + pn.scale_x_continuous(breaks=[0, 1, 2, 3, 4], labels=['S0', 'S1', 'S2', 'S3', 'S4'])
            )
            print(plot)
            df_he = df_he.merge(df_he.groupby('scene')['value'].sum().reset_index().rename(columns={'value': 'value_total'}))
            df_he['rate'] = df_he['value'] / df_he['value_total']
            return df_he
        elif pic_type == 'relative':
            df_he_s0 = df_he.query(f"scene=={div}").rename(columns={'value': 's1_value', 'low': 's1_low', 'up': 's1_up'}).copy()
            df_he_s0 = df_he.merge(df_he_s0.drop(columns='scene'), on=['class_'])
            df_he_s0['de_value'] = df_he_s0['value'] - df_he_s0['s1_value']
            df_he_s0['de_low'] = df_he_s0['low'] - df_he_s0['s1_low']
            df_he_s0['de_up'] = df_he_s0['up'] - df_he_s0['s1_up']
            df_he_s0['de_value_rate'] = df_he_s0['de_value'] / df_he_s0['s1_value']
            df_he_s0['de_value_rate'] = df_he_s0['de_value_rate'].map(lambda x: round(x, 3))
            
            plot = (
                pn.ggplot(df_he_s0.query("scene!=0"), pn.aes(x='scene', y='de_value_rate', group='class_'))
                + pn.geom_bar(pn.aes(fill='class_'), stat='identity', position=pn.position_dodge(width=0.7), width=0.5)
                + pn.geom_text(pn.aes(label='de_value_rate'), position=pn.position_dodge(width=0.7), size=8, show_legend=False, va='top')
                + pn.theme_bw()
                + pn.scale_x_continuous(breaks=[1, 2, 3, 4], labels=['S1', 'S2', 'S3', 'S4'])
            )
            print(plot)
            return df_he_s0
        else:
            raise ValueError("位置类型")
            
    def polar(self):
        df_g = self.df_heterogenity_g.copy()
        df_g_s0 = df_g.query("scene==0").copy().rename(columns={"value": 's0_value', 'low': 's0_low', 'up': 's0_up'}).drop(columns='scene')
        df_g = df_g.merge(df_g_s0)
        df_g['descend_value'] = df_g['value'] - df_g['s0_value']
        df_g['descend_up'] = df_g['up'] - df_g['s0_up']
        df_g['descend_low'] = df_g['low'] - df_g['s0_low']
        df_g_3 = df_g.query("scene==3").copy()
        df_g_3['descend_rate_value'] = (df_g_3['descend_value'] / df_g_3['s0_value'] * 100).map(lambda x: round(np.abs(x), 2))
        class_ = ['female', 'male', 'town', 'rural', 'age_0_14', 'age_15_64', 'age_65']
        data = {'IHD': [], 'COPD': [], 'LC': [], 'Stroke': []}
        for i in ['IHD', 'COPD', 'LC', 'Stroke']:
            for c in class_:
                data[i].append(df_g_3.query(f"disease_type=='{i}' and class_=='{c}'")['descend_rate_value'].values[0])
        print(data)
        pic_polar = (
            Polar()
            .add_schema(
                angleaxis_opts=opts.AngleAxisOpts(
                    data=class_, 
                    type_="category", 
                    splitline_opts=opts.SplitLineOpts(
                        is_show=True,
                        linestyle_opts=opts.LineStyleOpts(
                            width=1,
                            color='black'
                        )
                    ),
                ),
                radiusaxis_opts=opts.RadiusAxisOpts(splitarea_opts=opts.SplitAreaOpts(areastyle_opts=opts.AreaStyleOpts(opacity=0.5)))
            )
            .add("Stroke", data['Stroke'], type_="bar")
            .add("LC", data['LC'], type_="bar")
            .add("COPD", data['COPD'], type_="bar")
            .add("IHD", data['IHD'], type_="bar")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=True))
        )
        return pic_polar
    
    def decay_death_pm(self, SCENE, type_):
        """
        type_:pm或death
        """
        df_city_death = pd.DataFrame({})
        for scene in range(6):
            for disease_type in ['IHD', 'Stroke', 'COPD', 'LC']:
                df_temp = self.df_scene.filter(regex=f"scene_{scene}_{disease_type}.*?").copy()
                df_temp['city'] = self.df_scene['city']
                df_temp['scene'] = f'S{scene}'
                df_temp = df_temp.groupby(['city', 'scene']).sum().reset_index()
                df_temp['disease_type'] = disease_type
                df_temp.rename(columns={f"scene_{scene}_{disease_type}_mean": 'value', f"scene_{scene}_{disease_type}_low": 'low', f"scene_{scene}_{disease_type}_high": 'up'}, inplace=True)
                df_city_death = pd.concat([df_city_death, df_temp])
        df_city_death_g = df_city_death.groupby(['city', 'scene'])['value'].sum().reset_index()
        df_city_pm = pd.DataFrame({})
        for scene in range(6):
            for disease_type in ['IHD', 'Stroke', 'COPD', 'LC']:
                df_temp = self.df_scene.filter(regex=f"scene_{scene}_PM25_{disease_type}.*?").copy()
                df_temp['city'] = self.df_scene['city']
                df_temp['scene'] = f'S{scene}'
                df_temp = df_temp.groupby(['city', 'scene']).sum().reset_index()
                df_temp['disease_type'] = disease_type
                df_temp.rename(columns={f"scene_{scene}_PM25_{disease_type}_mean": 'value', f"scene_{scene}_PM25_{disease_type}_low": 'low', f"scene_{scene}_PM25_{disease_type}_high": 'up'}, inplace=True)
                df_city_pm = pd.concat([df_city_pm, df_temp])
        df_city_pm_g = df_city_pm.groupby(['city', 'scene'])['value'].mean().reset_index()
        if type_ == 'pm':
            df_v = df_city_pm_g.query(f"scene=='{SCENE}'")
            min_, max_ = df_city_pm_g['value'].min(), df_city_pm_g['value'].max()
        else:
            df_v = df_city_death_g.query(f"scene=='{SCENE}'")
            min_, max_ = 0,10000
        data = [[i[0], i[1]] for i in zip(df_v['city'], df_v['value'])]
        dist = (
            Map(init_opts=opts.InitOpts(width='500px', height='300px'))
            .add('', data, "china-cities", label_opts=opts.LabelOpts(is_show=False), is_map_symbol_show=False)
            .set_global_opts(
                visualmap_opts=opts.VisualMapOpts(
                    min_=min_,
                    max_=max_,
                    pos_left='11%',
                    pos_bottom='7%',
                    item_height=60,
                    item_width=5,
                    textstyle_opts=opts.TextStyleOpts(
                        font_size=18
                    )
                ),
                toolbox_opts=opts.ToolboxOpts(
                    feature=opts.ToolBoxFeatureOpts(
                        save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                            pixel_ratio=5,
                            background_color='white'
                        )
                    )
                )
            )
        )
        df_mean_pm = df_city_pm_g.groupby('scene')['value'].mean().reset_index()
        df_mean_pm['value'] = df_mean_pm['value'].map(lambda x: round(x, 1))
        plot = (
            pn.ggplot(df_mean_pm)
            + pn.geom_bar(pn.aes(x='scene', y='value'), stat='identity', width=0.6, fill='#FFA500')
            + pn.geom_text(pn.aes(x='scene', y='value', label='value'), va='bottom')
            + pn.theme_bw()
        )
        #plot.save("论文写作/论文图/图2/pm.png")
        #print(plot)
        grid = Grid(init_opts=opts.InitOpts(width='500px', height='300px'))
        grid.add(dist, 
                 grid_opts=opts.GridOpts(
                     pos_left='10%',
                     pos_right='10%',
                     pos_top='10%',
                     pos_bottom='10%'
                 )
                )
        return dist


def gini_coef(data):
    """
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths)-1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    """
    base = 0
    q_i_1 = 0
    p_i_1 = 0
    for i in zip(data['p_population'], data['p_sum_info_value']):
        base += (i[1] + q_i_1) * (i[0] - p_i_1)
        q_i_1 = i[1]
        p_i_1 = i[0]
    return 1 - base


def lorenz_curve(data):
    df_p_info = data[['population_y', 'sum_info_value']].copy().sort_values(by='sum_info_value').rename(columns={'population_y': 'population'})
    total_people = df_p_info['population'].sum()
    total_info = df_p_info['sum_info_value'].sum()
    df_p_info['p_population'] = df_p_info['population'].cumsum() / total_people
    df_p_info['p_sum_info_value'] = df_p_info['sum_info_value'].cumsum() / total_info
    line = (
        Line(init_opts=opts.InitOpts(width='600px', height='600px'))
        .add_xaxis(list(df_p_info['p_population'].map(lambda x: round(x, 3))))
        .add_yaxis('', list(df_p_info['p_sum_info_value'].map(lambda x: round(x, 3))), 
                   areastyle_opts=opts.AreaStyleOpts(opacity=0.5), is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=2, color='red'))
        .add_yaxis('Cumsum Proportion of Environment Risk Info (%)', 
                   list(df_p_info['p_population'].map(lambda x: round(x, 3))), 
                   areastyle_opts=opts.AreaStyleOpts(opacity=0.1), 
                   is_symbol_show=False)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(xaxis_opts=opts.AxisOpts(axispointer_opts=opts.AxisPointerOpts(label=opts.LabelOpts('Cumsum Proportion of People (%)'), is_show=True), type_='value', interval=0.1))
    )
    gini = gini_coef(df_p_info)
    print(gini)
    #print(list(df_p_info['p_sum_info_value'].map(lambda x: round(x, 3))))
    return line

def pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s.capitalize()

def inequality(data):
    df_point = data[['info_index','death_reduce_value','death_reduce_rate', 'city', 'log_sum_info_value_mean', 'log_death_rate']].copy()
    df_point['death_reduce_rate'] =  (-df_point['death_reduce_rate'] * 100).astype(int)
    df_point['info_index'] = np.log(1 + df_point['info_index'])
    df_point['death_reduce_value'] = np.log(1 - df_point['death_reduce_value'])
    df_point = df_point.query("log_sum_info_value_mean>3")
    df_high = df_point.query("death_reduce_rate>8").copy()
    df_high['city'] = df_high['city'].map(pinyin)
    print(df_high.shape)
    plot = (
        pn.ggplot(df_point, pn.aes(x='log_sum_info_value_mean', y='log_death_rate'))
        + pn.geom_point(pn.aes(size='death_reduce_rate', color='death_reduce_rate'))
        + pn.geom_smooth(method='lm',level=0.95, colour="gray")
        + pn.geom_text(df_high, pn.aes(x='log_sum_info_value_mean', y='log_death_rate', label='city'), size=16, nudge_y=0.03, nudge_x=-0.03)
        + pn.xlab("Mean Info(log)")
        + pn.ylab("Decrease Death/10000(log)")
        + pn.guides(color=pn.guide_legend(title="DDR(%)"), size=pn.guide_legend(title="DDR(%)"))
        + pn.theme_bw()
        + pn.theme(text=pn.element_text(family=['SimHei', 'Arial Unicode MS']), 
                   figure_size=(8, 8), 
                   axis_title_x=pn.element_text(vjust=1, size=20, face='bold'),
                   axis_title_y=pn.element_text(hjust=2, size=20, face='bold'),
                   axis_text_x=pn.element_text(hjust=2, size=18, face='bold'),
                   axis_text_y=pn.element_text(vjust=4, size=18, face='bold'),
                   legend_text=pn.element_text(size=18, face='bold'),
                   legend_title=pn.element_text(size=18, face='bold'),
                  ) 
    )
    plot.save("result/regression_pic.png")
    print(plot)
    

def draw_ques(data):
    df_ques = data.copy()
    df_ques['outdoor_time'] = df_ques['outdoor_time'] / 24
    df_ques['haze_attention'] = df_ques['haze_attention'].map(lambda x: 0 if x <= 3 else 1)
    df_ques_pattern = (
        df_ques[['province', 'outdoor_time','haze_perception','haze_cognition','haze_attention']]
        .groupby('province').mean().reset_index()
        .set_index('province')
        .stack()
        .reset_index()
        .rename(columns={'level_1': 'pattern', 0: 'value_pattern'})
    )
    df_ques_action = (
        df_ques[['province', 'mask_ratio','cancel_outdoor','risk_action','work_airclean','home_airclean']]
        .groupby('province').mean().reset_index()
        .set_index('province')
        .stack()
        .reset_index()
        .rename(columns={'level_1': 'action', 0: 'value_action'})
    )
    use_sample = df_ques.groupby('province')['city'].count().reset_index().query("city>=20")[['province']]
    df_ques_pattern = df_ques_pattern.merge(use_sample)
    df_ques_action = df_ques_action.merge(use_sample)
    plot_pattern = (
        pn.ggplot()
        + pn.geom_bar(df_ques_pattern, pn.aes(x='province', y='value_pattern'), stat='identity', fill='#1E90FF', width=0.5)
        + pn.facet_wrap('~pattern', nrow=1, scales='free_x')
        + pn.coord_flip()
        + pn.theme_bw()
        + pn.theme(text=pn.element_text(family=['SimHei', 'Arial Unicode MS']), figure_size=(10, 10)) 
    )
    #print(plot_pattern)
    
    plot_action = (
        pn.ggplot()
        + pn.geom_bar(df_ques_action, pn.aes(x='province', y='value_action'), stat='identity', width=0.5, fill='#006400')
        + pn.facet_wrap('~action', nrow=1, scales='free_x')
        + pn.coord_flip()
        + pn.theme_bw()
        + pn.theme(text=pn.element_text(family=['SimHei', 'Arial Unicode MS']), figure_size=(10, 10)) 
    )
    #print(plot_action)
    
    pattern_action = []
    for i in ['haze_perception','haze_cognition','haze_attention']:
        for j in ['mask_ratio','cancel_outdoor','work_airclean','home_airclean']:
            pattern_action.append([i, j])
    df_pattern_action = pd.DataFrame(pattern_action, columns=['pattern', 'action'])
    df_pattern_action = df_pattern_action.merge(df_ques_pattern).merge(df_ques_action)
    plot_pattern_action = (
        pn.ggplot(df_pattern_action, pn.aes(x='value_action', y='value_pattern'))
        + pn.geom_point()
        + pn.stat_smooth(method='lm', fill='#3CB371')
        + pn.facet_grid('action~pattern', scales='free')
        + pn.coord_flip()
        + pn.theme_bw()
        + pn.theme(text=pn.element_text(family=['SimHei', 'Arial Unicode MS'], size=12), figure_size=(10, 10)) 
    )
    print(plot_pattern_action)
    return df_pattern_action

def draw_pred(data):
    df_cancel_od = pd.read_csv("data/cancel_outdoor_pred_rf.csv")
    df_use = data[['city', 'pred_outdoor_time', 'pred_mask_ratio', 'pred_cancel_outdoor', 'pred_work_airclean', 'pred_home_airclean', 'pred_haze_attention']].copy()
    df_use['pred_air_cleaner'] = (df_use['pred_work_airclean'] + df_use['pred_home_airclean']) / 2
    df_use['pred_air_cleaner'] = df_use['pred_air_cleaner'].map(lambda x: 0.85 if x>=1 else x)
    df_use['pred_mask_ratio'] = df_use['pred_mask_ratio'].map(lambda x: 0.9 if x>=1 else x)
    df_use['pred_outdoor_time'] = df_use['pred_outdoor_time'] / 24
    df_use.drop(columns=['pred_work_airclean', 'pred_home_airclean', 'pred_cancel_outdoor'], inplace=True)
    df_use = df_use.merge(df_cancel_od)
    df_use = df_use.set_index('city').stack().reset_index()
    df_use.columns = ['city', 'params', 'value']
    df_use.replace({'pred_outdoor_time': 'ODR', 'pred_mask_ratio': 'MR', 'pred_cancel_outdoor': 'CODR', 'pred_air_cleaner': 'ACR', 'pred_haze_attention': 'ATTR'}, inplace=True)
    plot = (
        pn.ggplot(df_use, pn.aes(x='value', fill='params', color='params'))
        + pn.geom_density()
        + pn.scale_x_continuous(limits=[0, 1])
        + pn.theme_bw()
    )
    plot.save("result/预测数据概率分布.png")
    print(plot)
    return None
