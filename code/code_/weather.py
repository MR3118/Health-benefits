import pandas as pd
import glob
import os
from code_.utils import FlexibleTime
import numpy as np

class Weather:
    
    def __init__(self, file, type_, time_format, check_type=True):
        """可输入AQI所在文件目录或者AQI数据
        time_format: 支持聚合为date、month、year、hour
        type_: 要抽取的空气污染类型，可同时输入多个，必须是list类型
        file: file: 当合并原始数据分散文件时，只能是数据的根目录，当是已经处理好的数据时，必须为dataframe
        """
        self.fpath = None
        self.file = None
        self.type_ = type_
        if check_type:
            all_types = ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO', 'PM2.5_24h']
            for i in self.type_:
                if i not in all_types:
                    raise KeyError(f"所输入的污染物{i}不属于{all_types}中的任何一种或24h污染物种类，请仔细检查")
                
        self.time_format = time_format
        assert isinstance(self.type_, list)
        if isinstance(file, str):
            """当file为文件目录时，用于处理原始数据"""
            self.fpath = file
        else:
            """当file为dataframe时"""
            self.df = file.copy()
            
    def gather_files(self):
        if not os.path.isdir(self.fpath):
            raise TypeError(f"路径错误: {self.fpath}")
        files = os.listdir(self.fpath)
        file_path  = [self.fpath + f'/{i}' for i in files]
        df_full = pd.DataFrame({})
        if os.path.isfile(file_path[0]):
            for file in file_path:
                df_temp = pd.read_csv(file)
                df_full = pd.concat([df_full, df_temp])
        else:
            for path in file_path:
                file_list = glob.glob(f"{path}/*.csv")
                for file in file_list:
                    df_temp = pd.read_csv(file)
                    df_full = pd.concat([df_full, df_temp])
        self.df = df_full
        return self
    
    def transform(self, data_type='mean', date_span=None, check_data=True):
        """
        data_type: 聚合方式：mean, min, max
        type_: 获取污染类型AQI、pm2.5等
        date_span: 日期范围: 例如[20200101， 20201231]
        """
        print("""提示：国控站点城市空气污染数据从2014年5月13日开始，2015年1月份数据缺失，从2015年2月起，共有366城市数据，少量城市数据缺失，2014年仅包含189个城市""")
        if self.type_ is not None:
            self.df = self.df.query(f"type in {self.type_}")
        if date_span is not None:
            if not isinstance(date_span, list):
                raise TypeError("date_span参数必须为2个值的列表形式，例如[20150301, 20190302]")
            self.df = self.df.query(f"{date_span[0]}<=date<={date_span[1]}")
        self.df = (
            self.df.set_index(['date', 'hour', 'type'])
            .stack()
            .reset_index()
            .groupby(['date', 'hour', 'level_3', 'type'])
            .mean()
            .unstack()
            .reset_index()
        )
        columns = self.df.columns
        new_columns = []
        for i in columns:
            if i[0] != 0:
                new_columns.append(i[0])
            else:
                new_columns.append(i[1])
        self.df.columns = new_columns
        self.df.rename(columns={'level_3': 'city'}, inplace=True)
        
        if self.time_format == "hour":
            df_g = self.df.groupby(['date', 'city', 'hour'])
        elif self.time_format == "date":
            self.df.drop(columns='hour', inplace=True)
            df_g = self.df.groupby(['date', 'city'])
        elif self.time_format == 'month':
            self.df.drop(columns='hour', inplace=True)
            self.df['month'] = self.df['date'].map(lambda x: int(str(x)[:6]))
            self.df.drop(columns='date', inplace=True)
            df_g = self.df.groupby(['month', 'city'])
        elif self.time_format == 'year':
            self.df.drop(columns='hour', inplace=True)
            self.df['year'] = self.df['date'].map(lambda x: int(str(x)[:4]))
            self.df.drop(columns='date', inplace=True)
            df_g = self.df.groupby(['year', 'city'])
        else:
            raise TypeError(f"未知的时间类型{self.time_format}")
        
        if data_type == 'mean':
            self.df = df_g.mean().reset_index()
        elif data_type == 'min':
            self.df = df_g.min().reset_index()
        elif data_type == 'max':
            self.df = df_g.max().reset_index()
        else:
            raise TypeError(f"未知的聚合类型{data_type}")
        print('sss', self.df.shape)
            
        if check_data:
            self.check()
            
            
        return self
    
    def check(self, check_time=True, check_city=True, check_na=True):
        """验证数据是否是完整的面板数据"""
        # check time
        if check_time:
            time_min, time_max = self.df[self.time_format].min(), self.df[self.time_format].max()
            real_time_num = len(set(self.df[self.time_format]))
            ideal_date_num = (FlexibleTime(time_max) - FlexibleTime(time_min)).span
            if real_time_num != ideal_date_num:
                print(f"警告：实际天数与应该天数不相等, 实际：{real_time_num}, 理想：{ideal_date_num}，考虑是否填充缺失值")
            else:
                print(f"未发现时间异常")
        # check city
        if check_city:
            real_city_num = len(set(self.df['city']))
            df_check_g = self.df.groupby(self.time_format)['city'].count().reset_index()
            city_max_num, city_min_num = df_check_g['city'].max(), df_check_g['city'].min()
            if city_max_num != city_min_num:
                print(f"警告：不同月份含有的城市数量不同, 最小：{city_min_num}, 最大：{city_max_num}，总计：{real_city_num}考虑是否填充缺失值")
            else:
                print(f"未发现城市异常")
        
        # check N/A
        if check_na:
            columns = self.df.columns
            self.na_columns = {}
            len_df = self.df.shape[0]
            for i in columns:
                na_num = pd.isnull(self.df[i]).sum()
                if na_num != 0:
                    self.na_columns[i] = {'na_num': na_num, 'ratio': na_num / len_df}
            if len(self.na_columns) != 0:
                print("发现缺失值：\n", self.na_columns)
            else:
                print("未发现缺失值")
        return self
            
    def fill_index(self):
        # 线性插值
        cities = list(set(self.df['city']))
        time_min, time_max = self.df[self.time_format].min(), self.df[self.time_format].max()
        time_list = (FlexibleTime(time_max) - FlexibleTime(time_min)).get_range()
        new_index = []
        for i in time_list:
            for j in cities:
                new_index.append([i, j])
        df_new = pd.DataFrame(new_index, columns=[self.time_format, 'city'])
        if not isinstance(df_new.loc[0, self.time_format], type(self.df.loc[0, self.time_format])):
            self.df[self.time_format] = self.df[self.time_format].astype(type(df_new.loc[0, self.time_format]))
        df_fill_na = pd.DataFrame(new_index, columns=[self.time_format, 'city'])
        for i in self.type_:
            df_type = self.df[[self.time_format, 'city', i]]
            if df_type[i].dtype is np.dtype('O'):
                print(f"[fill_index]警告: 列'{i}'为object类型， 尝试转成float类型")
                df_type[i] = df_type[i].map(self._to_float)
            
            df_type = pd.merge(df_new, df_type, how='left')
            df_type = df_type.groupby([self.time_format, 'city'])[i].mean().unstack()
            for j in df_type.columns:
                # 大于5年没有数据，不补数据
                if pd.isnull(df_type.loc[df_type.index[0], j]) and pd.isnull(df_type.loc[df_type.index, j]).sum() > 0:
                    df_type.loc[df_type.index[0], j] = df_type.loc[df_type.index, j].min()
                if pd.isnull(df_type[j]).any():
                    df_type[j].interpolate(inplace=True)
            df_type = df_type.stack().reset_index()
            df_type.columns = [self.time_format, 'city', i]
            df_fill_na = pd.merge(df_fill_na, df_type, how='left', on=[self.time_format, 'city'])
        if pd.isnull(df_fill_na).any().sum() != 0:
            print("警告：仍然发现缺失值，请仔细检查")
            self.df = df_fill_na
            self.check()
            
        self.df = df_fill_na
        return self
    
    @staticmethod
    def _to_float(x):
        try:
            x = float(x)
        except ValueError:
            x = None
        return x