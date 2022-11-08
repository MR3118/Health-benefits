import pandas as pd
import re
import os
import datetime
import json
import tabulate as tb
import numpy as np
from IPython.display import HTML, display

class ReadData:
    
    def __init__(self, file_path, *args, **kwargs):
        self.fpath = file_path
        
        try:
            file_type = re.search(r".*?\.(.*?)$", file_path)[1]
        except Exception as e:
            file_type = None
            raise TypeError(f"未识别的文件类型{file_type}")
        self.file_type = file_type
        self.args = args
        self.kwargs = kwargs
        
    def infer_csv_format(self):
        if 'encoding' not in self.kwargs:
            encoding_list = ['utf8', 'gbk','utf-8-sig','GB2312','gb18030']
            encoding = 'utf8'
            for i in encoding_list:
                try:
                    pd.read_csv(self.fpath, encoding=i, nrows=10)
                    encoding = i
                    break
                except Exception as e:
                    print(f'尝试编码{i}读取错误', e)
                    encoding = None
                    continue
            if encoding is None:
                raise TypeError("未知encoding, 不在'utf-8', 'gbk','utf-8-sig','GB2312','gb18030'之中")
            self.kwargs['encoding'] = encoding
                
        if 'sep' not in self.kwargs:
            sep_list = ['\t', ',', '\s+']
            sep = None
            for i in sep_list:
                df_temp = pd.read_csv(self.fpath, sep=i, nrows=0)
                if len(df_temp.columns) > 1:
                    sep = i
                    break
            if sep is None:
                print("未知sep, 不在'\t', ',', '\s+'之中，或文件只有一列, 将使用默认分隔符")
                sep = ','
            self.kwargs['sep'] = sep
        
    def read(self):
        if self.file_type in ['csv', 'txt']:
            self.infer_csv_format()
            _df_result = pd.read_csv(self.fpath, **self.kwargs)
        elif self.file_type in ['xls', 'xlsx']:
            _df_result = pd.read_excel(self.fpath, **self.kwargs)
        else:
            raise TypeError("未知文件类型")
        return _df_result
    
    
class FlexibleTime:
    """
    灵活时间运算，主要计算连个时间之间的包含的天数（闭区间）,以及计算出之间的每天时间值
    输入的时间必须为“%Y%m%d”格式的整型或字符串，或者是datetime时间数据类型
    例如：201901，20190101，2019，"20100101", datetime.datetime(2020, 1, 2)
    """
    
    def __init__(self, date):
        if isinstance(date, int) or isinstance(date, str) or isinstance(date, float):
            date = str(int(date))
            self.span = None
            self.other_date = None
            len_date = len(date)
            if len_date == 4:
                self.format_ = "%Y"
                self.date = datetime.datetime.strptime(date, self.format_)
            elif len_date == 6:
                self.format_ = "%Y%m"
                self.date = datetime.datetime.strptime(date, self.format_)
            elif len_date == 8:
                self.format_ = "%Y%m%d"
                self.date = datetime.datetime.strptime(date, self.format_)
            else:
                raise TypeError(f"暂不支持的时间格式{date}")
        elif isinstance(datetime.datetime(2020, 12, 1), datetime.datetime):
            self.date = date
            self.format_ = "%Y%m%d"
        else:
            raise TypeError(f"暂不支持的时间格式{date}")
            
    def __sub__(self, other):
        assert self.format_ == other.format_
        assert other.date < self.date
        self.other_date = other.date
        if self.format_ == "%Y":
            self.span = self.date.year - other.date.year + 1
        elif self.format_ == "%Y%m":
            self.span = (self.date.year - other.date.year - 1) * 12 + self.date.month + 12 - other.date.month + 1
        elif self.format_ == "%Y%m%d":
            self.span = (self.date - other.date).days + 1
        else:
            raise TypeError("未知时间类型")
        return self
    
    def get_range(self):
        if self.span == None:
            raise TypeError("必须两个FlexibleTime类型数据相减后才能使用该函数")
        time_span = []
        if self.format_ == "%Y":
            other_date = int(datetime.datetime.strftime(self.other_date, self.format_))
            for i in range(self.span):
                time_span.append(other_date + i)
        elif self.format_ == "%Y%m":
            year = self.other_date.year
            month = self.other_date.month
            for i in range(self.span):
                time_span.append(int(year * 100 + month))
                month += 1
                if month > 12:
                    year += 1
                    month = 1
        elif self.format_ == "%Y%m%d":
            other_date = self.other_date
            for i in range(self.span):
                time_span.append(int(datetime.datetime.strftime(other_date, self.format_)))
                other_date = other_date + datetime.timedelta(days=1)
        else:
            raise TypeError("未知时间类型")
        time_span = [str(i) for i in time_span]
        
        return time_span
    
    
class ChinaRegion:
    
    def __init__(self):
        with open("code_/dist.json", 'r') as f:
            dist = json.load(f)
        self.dist = dist
        self.cities = list(dist['city2province'].keys())
        self.provinces = list(dist['province2city'].keys())
        self.county = list(dist['county2city'])
        self.all_loc = self.cities + self.provinces
    
    def simple_location(self, x, level='city'):
        if level == 'city':
            locs = self.cities
        elif level == 'province':
            locs = self.provinces
        elif level == 'county':
            locs = self.county
        else:
            raise ValueError("只能是province或city或county")
        real = []
        city = None
        if isinstance(x, str):
            for i in locs:
                if i in x:
                    real.append(i)
            if len(real) > 1:
                length = 0
                for j in real:
                    if len(j) > length:
                        city = j
            elif len(real) == 0:
                try:
                    if city[-1:] == '市':
                        print('未识别市', x)
                except Exception as e:
                    pass
            else:
                city = real[0].strip()
        else:
            pass
        return city
    
    
    def province_of_city(self, city):
        try:
            province = self.dist['city2province'][city]
        except Exception as e:
            province = None
        return province
    
    def city_of_province(self, province):
        try:
            city = self.dist['province2city'][province]
        except Exception:
            city = None
        return city
    
    def city_of_county(self, county):
        try:
            city = self.dist['county2city'][county]
        except Exception as e:
            city = None
        return city
    
    def county_of_city(self, city):
        try:
            county = self.dist['city2county'][city]
        except Exception:
            county = None
        return county
    
    @staticmethod
    def location_of_context(x, loc):
        x = str(x)
        location = None
        for i in loc:
            if i in x:
                location = i
                break
        return location
    
    @staticmethod
    def is_north_province(x):
        north = ['黑龙江', '吉林', '辽宁', '北京', '天津', '河北', '山东', '河南', '山西', '陕西', '内蒙古']
        south = ['江苏','安徽','浙江','上海','湖北','湖南','江西','福建','云南','贵州','四川','重庆','广西','广东','海南']
        if x in north:
            return 1
        elif x in south:
            return 0
        else:
            return None

    def is_north_city(self, x):
        try:
            province = self.province_of_city(x)
            is_north = self.is_north_province(province)
        except Exception as e:
            is_north = None
        return is_north
    
    
class AutoData:
    
    def __init__(self):
        """数据集中管理
        适用于数据过多，数据信息重复性手动获取工作量大，将数据统一管理
        数据结构
        |—— AutoData
        |   |—— data_name
        |   |   |—— file_path
        |   |   |—— shape
        |   |   |—— columns
        |   |   |—— comments
        |   |   |—— NA_info
        |   |   |—— describe
        |   |   |   |—— object_info
        |   |   |   |—— digital_info
        |   |   |—— main_keys
        |   |   |   |—— key_1
        |   |   |   |   |—— min
        |   |   |   |   |—— max
        |   |   |   |   |—— unique_count
        |   |   |   |   |—— key_type {country, province, city, county, ymdhm, ymdh, ymd, ym, year, month, day, hour, minute, uid, record_id}
        |   |   |—— gene
        |   |   |   |—— dataset_1
        |   |   |   |   |—— generation
        |   |   |   |   |—— join_data_1
        |   |   |   |   |   |—— usecols
        |   |   |   |   |   |—— on
        |   |   |   |   |   |—— where
        |   |   |   |   |   |—— out
        |   |   |   |   |—— join_data_2
        |   |   |   |   |   |—— usecols
        |   |   |   |   |   |—— on
        |   |   |   |   |   |—— where
        |   |   |   |   |   |—— out
        |   |   |   |—— dataset_2
        |   |   |—— register_time
        """
        if os.path.exists("data/AutoData.json"):
            with open("data/AutoData.json", 'r') as f:
                self.dataset = json.load(f)
        else:
            if not os.path.exists("data"):
                os.mkdir("data")
            self.dataset = {}
        self.base_info = ['file_path', 'shape', 'comments', 'columns', 'register_time']
        self.detail_info = ['NA_info', 'describe']
        self.merge_info = ['main_keys']
            
    def register(self, fpath, data_name, data=None, comments=None, main_keys=None):
        """
        :param fpath str, 文件路径
        :param data_name str, 文件名字
        :param data pandas.DataFrame, 文件名字
        :param comments str, 数据描述信息，备注等，可选
        :param main_keys list[{"key": 'name', "type": 'key_type'}, {...}, ...], 主要用于和其他表关联的字段，可选
        """
        if data is None:
            data = ReadData(fpath).read()
        else:
            data = data
        self.dataset[data_name] = {}
        self.dataset[data_name]['file_path'] = fpath
        self.dataset[data_name]['shape'] = [data.shape[0], data.shape[1]]
        self.dataset[data_name]['columns'] = list(data.columns)
        if comments is not None:
            assert comments.__class__ == str
            self.dataset[data_name]['comments'] = comments
        else:
            self.dataset[data_name]['comments'] = "暂无备注"
        na_count = []
        na_rate = []
        for i in data.columns:
            na_num = int(pd.isnull(data[i]).sum())
            na_ratio = round(na_num / data.shape[0], 4)
            na_count.append(int(na_num))
            na_rate.append(float(na_ratio))
        self.dataset[data_name]['NA_info'] = {"header": ['index'] + list(data.columns), "count": ['na_count'] + na_count, "rate": ['na_rate'] + na_rate}
        
        object_cols = []
        digital_cols = []
        dtypes = data.dtypes.reset_index()
        for i in zip(dtypes['index'], dtypes[0]):
            if i[1] == np.dtype("O"):
                object_cols.append(i[0])
            else:
                digital_cols.append(i[0])
                
        if len(object_cols) > 0:
            object_desc = data[object_cols].describe().reset_index().to_dict("list")
            for i in object_desc.keys():
                object_desc[i] = [round(float(j), 4) if not isinstance(j, str) and not pd.isnull(j)  else j for j in object_desc[i]]
                object_desc[i] = [None if pd.isnull(j) else j for j in object_desc[i]]
                
        else:
            object_desc = None
            
        if len(digital_cols) > 0:
            digital_desc = data[digital_cols].describe().reset_index().to_dict("list")
            for i in digital_desc.keys():
                digital_desc[i] = [round(float(j), 4) if not isinstance(j, str) and pd.isnull(j) else j for j in digital_desc[i]]
                digital_desc[i] = [None if pd.isnull(j) else j for j in digital_desc[i]]
                
        else:
            digital_desc = None
            
        self.dataset[data_name]['describe'] = {}
        self.dataset[data_name]['describe'] = {"object_info": object_desc, "digital_info": digital_desc}
        
        main_key_info = [['key', 'min', 'max', 'distinct', 'type']]
        if main_keys is not None:
            for i in main_keys:
                min_ = data[i['key']].min()
                max_ = data[i['key']].max()
                uni = len(set(data[i['key']]))
                type_ = i['type']
                main_key_info.append([i['key']] + [min_, max_, uni, type_])
        self.dataset[data_name]['main_keys'] = main_key_info
        self.dataset[data_name]['register_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("data/AutoData.json", 'w') as f:
            json.dump(self.dataset, f)
                
    def get_info(self, data_name=None, level='base'):
        """
        :param data_name str, 数据名称，不指定时默认返回所有数据
        :param level str, 返回指定级别信息，可为:base, detail, merge
        :param field str, 返回指定字段信息
        """
        if data_name is None:
            root_keys = self.dataset.keys()
        else:
            root_keys = [data_name]
        base_table = []
        base_header = ['data_name'] + self.base_info
        for key in root_keys:
            data = self.dataset[key]
            if level == 'base':
                base_keys = [key]
                for i in self.base_info:
                    if i in data.keys():
                        base_keys.append(data[i])
                    else:
                        base_keys.append(None)
                base_table.append(base_keys)
            elif level == 'detail':
                print(f"****数据集：{key}*****")
                detail_table = []
                header = []
                for col in data['describe']['object_info']:
                    detail_table.append(data['describe']['object_info'][col] + ['——' for i in range(4)])
                    header.append(col)
                for col in data['describe']['digital_info']:
                    detail_table.append(data['describe']['digital_info'][col])
                    header.append(col)
                detail_table = np.array(detail_table).T.tolist()
                na_info = pd.DataFrame([data['NA_info']['count'], data['NA_info']['rate']], columns=data['NA_info']['header'])
                detail_table.append([na_info.loc[0, i] for i in header])
                detail_table.append([na_info.loc[1, i] for i in header])
                display(HTML(tb.tabulate(detail_table, headers=header, tablefmt='html')))
            elif level == 'merge':
                print(f"****数据集：{key}*****")
                display(HTML(tb.tabulate(data['main_keys'][1:], headers=data['main_keys'][0], tablefmt='html')))
            else:
                raise KeyError(f"未知信息参数: {level}")
        if len(base_table) > 0:
            display(HTML(tb.tabulate(base_table, headers=base_header, tablefmt='html')))
            
    def set_info(self, data_name, field, value):
        """更改已有字段属性
        :param data_name str, 数据集名称
        :param field str, 字段名称
        :param value str, 更改后的属性值
        """
        yes = input(f"是否改变参数{data_name}:{field}，是请输入y,否请输入n:")
        if yes == 'y':
            self.dataset[data_name][field] = value
            with open("data/AutoData.json", 'w') as f:
                json.dump(self.dataset, f)
            print("参数已设置保存成功")
        elif yes == 'n':
            pass
        else:
            raise ValueEerror("输入格式错误！请重新运行，输入y或n")
            
    def delete(self, data_name):
        """删除某个数据集
        """
        yes = input(f"是否删除数据集 {data_name}，是请输入y,否请输入n:")
        if yes == 'y':
            del self.dataset[data_name]
            with open("data/AutoData.json", 'w') as f:
                json.dump(self.dataset, f)
            print("删除并保存成功")
        elif yes == 'n':
            pass
        else:
            raise ValueEerror("输入格式错误！请重新运行，输入y或n")
        
