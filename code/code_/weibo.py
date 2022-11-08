import pandas as pd
import re
import numpy as np
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
try:
    import paddlehub as hub
except Exception:
    pass
from collections import defaultdict
from code_.utils import ChinaRegion
        
        
class Weibo:
    
    def __init__(self, dataframe):
        dataframe = dataframe.copy()
        self.df = dataframe
        
    @staticmethod
    def split_time(x, format_):
            _time = None
            if isinstance(x, str):
                try:
                    _date = x.split(' ')[0]
                    if len(_date) == 10:
                        if format_ == 'day':
                            _time = _date
                        elif format_ == 'month':
                            _time = _date[:7]
                        elif format_ == 'year':
                            _time = _date[:4]
                        else:
                            raise ValueError("不存在的时间类型")
                except Exception as e:
                    pass
            return _time
    
    @staticmethod
    def clean(x):
        if isinstance(x, str):
            x = x
        else:
            x = str(x)
        new = re.sub(r'[^\u4e00-\u9fa5]', '', x)
        return new
    
    def trim_time(self, time_col, time_format):
        """将时间处理为目标格式"""
        self.df[time_col] = self.df[time_col].map(lambda x: self.split_time(x, time_format))
        return self
        
    def trim_content(self, content_col):
        """除去内容中非中文字符"""
        self.df[content_col] = self.df[content_col].map(self.clean)
        return self
    
    """
    def gen_event_loc(self, content_col, chunksize=1000):
        len_df = len(self.df)
        chunk_num = len_df // chunksize
        new_col = []
        module = hub.Module(name="lac")
        for i in range(chunk_num):
            content = list(self.df[content_col][chunksize * i : chunksize * (i + 1)])
            try:
                cuts = module.cut(text=content, use_gpu=True, batch_size=500, return_tag=True)
            except TypeError:
                print(content)
                raise TypeError
            for arr in cuts:
                freq = defaultdict(int)
                likely_loc = None
                likely_loc_num = 0
                for i in zip(arr['word'], arr['tag']):
                    if i[1] == 'LOC':
                        freq[i[0]] += 1
                if len(freq) > 0:
                    for i in freq.keys():
                        if freq[i] > likely_loc_num:
                            likely_loc = i
                new_col.append(likely_loc)
        self.df['event_loc'] = new_col
        return self
    """
    def gen_emotion(self, content_col, chunksize=10000):
        lac = hub.Module(name='senta_bilstm')
        result = pd.DataFrame({})
        rows = self.df.shape[0]
        n_chunk = rows // chunksize
        for i in range(n_chunk + 1):
            chunk = list(self.df[content_col][i * chunksize: (i + 1) * chunksize])
            v = lac.sentiment_classify(chunk, use_gpu=True, batch_size=200)
            v = pd.DataFrame(v)[['sentiment_label', 'positive_probs']]
            result = pd.concat([result, v])
        self.df['sentiment_label'] = list(result['sentiment_label'])
        self.df['positive_probs'] = list(result['positive_probs'])
        return self

    
class UserInfo:
    
    def __init__(self, dataframe):
        self.df = dataframe
        self.cr = ChinaRegion()
        
    def loc_from_location(self, location_col, tail=None, prefix=None):
        p_name = 'province'
        c_name = 'city'
        if tail:
            p_name = 'province' + tail
            c_name = 'city' + tail
        if prefix:
            p_name = prefix + 'province'
            c_name = prefix + 'city'
        self.df[p_name] = self.df[location_col].map(lambda x: x.split(' ')[0] if isinstance(x, str) else None)
        self.df[c_name] = self.df[location_col].map(lambda x: x.split(' ')[-1] if isinstance(x, str) and len(x.split(' '))==2 and x.split(' ')[0]!= '海外' else None)
        self.df[c_name] = self.df[[c_name, p_name]].apply(lambda x: x[p_name] if x[p_name] in ['上海', '北京', '重庆', '天津'] else x[c_name], axis=1)
        self.df[p_name] = self.df[p_name].map(lambda x: None if x in ['海外', '其他'] else x)
        return self
    
    def loc_from_verified_reason(self, verified_reason_col):
        self.df['u_province'] = self.df[verified_reason_col].map(lambda x: self.cr.location_of_context(x, self.cr.provinces))
        self.df['u_city'] = self.df[verified_reason_col].map(lambda x: self.cr.location_of_context(x, self.cr.cities))
        self.df['u_county'] = self.df[verified_reason_col].map(lambda x: self.cr.location_of_context(x, self.cr.county))
        self.df['city'] = self.df[['u_county', 'city']].apply(lambda x: self.cr.city_of_county(x.u_county) if pd.isnull(x.city) and not pd.isnull(x.u_county) else x.city, axis=1)
        self.df['province'] = self.df[['province', 'city']].apply(lambda x: self.cr.province_of_city(x.city) if pd.isnull(x.province) and not pd.isnull(x.city) else x.province, axis=1)
        self.df['province'] = self.df[['province', 'u_province']].apply(lambda x: x.u_province if pd.isnull(x.province) and not pd.isnull(x.u_province) else x.province, axis=1)
        self.df['city'] = self.df[['city', 'u_city']].apply(lambda x: x.u_city if pd.isnull(x.city) and not pd.isnull(x.u_city) else x.city, axis=1)
        return self
    
    def city_from_checkin(self, checkin_col, city_col):
        self.df[city_col] = self.df[[city_col, checkin_col]].map(lambda x: x[checkin_col] if pd.isnull(x[city_col]) and not pd.isnull(x[checkin_col]) else x[city_col], axis=1)
        return self