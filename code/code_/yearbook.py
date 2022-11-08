import pandas as pd
import re
import glob
import os
import win32com.client as win32
from code_.utils import ChinaRegion
from code_.utils import FlexibleTime


class YearBook:
    
    def __init__(self):
        self.three_cols = ['city', '工业固体废物综合利用率','城镇生活污水集中处理率','生活垃圾无害化处理率']
        self.five_cols = ['city', '工业废水排放量(万吨)','工业废水排放达标量(万吨)','工业二氧化硫去除量(吨)','工业二氧化硫排放量(吨)','工业烟尘去除量(吨)','工业烟尘排放量(吨)']
    
    def raw_year(self, root_dir):
        """
        2005,
        """
        files = glob.glob(root_dir + '\\*.xls')
        for f in files:
            # 解老xls视图锁定问题
            wb = win32.DispatchEx('Excel.Application').Workbooks.Open(f)
            wb.SaveAs(f + 'x', FileFormat=51)
            wb.close()
            df_wb = pd.read_excel(f + 'x')
            df_wb = df.dropna(subset=[list(df.columns)[0]])
    
    def m_2005(self, path):
        """
        2005, 2006: 2-42, 2-43
        """
        df_m = None
        for i in ['2-42', '2-43']:
            if i == '2-42':
                skip = 4
            else:
                skip = 5
            df_wb = pd.read_excel(path, sheet_name=i, skiprows=skip)
            df_wb = df_wb.drop(columns=['Unnamed: 0', '城　市'])
            cols = list(df_wb.columns)
            cols[0] = 'city'
            df_wb.columns = cols
            if df_m is None:
                df_m = df_wb
            else:
                df_m = pd.merge(df_m, df_wb)
        df_m['year'] = 2005
        df_m.drop_duplicates(subset='city', inplace=True)
        return df_m
        
    def m_2006(self, path):
        """
        2005, 2006: 2-42, 2-43
        """
        df_m = None
        for i in ['2-42', '2-43']:
            if i == '2-42':
                columns = self.five_cols
            else:
                columns = self.three_cols
            df_wb = pd.read_excel(path, sheet_name=i, skiprows=4)
            df_wb = df_wb.dropna(subset=['城市'])
            df_wb.columns = columns
            if df_m is None:
                df_m = df_wb
            else:
                df_m = pd.merge(df_m, df_wb)
        df_m['year'] = 2006
        df_m.drop_duplicates(subset='city', inplace=True)
        return df_m
    
    def m_2007(self, root_dir):
        files = glob.glob(root_dir + '\\*.xls')
        df_m = None
        for f in files:
            path = os.getcwd() + '\\' + f
            path = path.replace('/', '\\')
            if os.path.exsists(path):
                pass
            else:
                wb=win32.DispatchEx('Excel.Application').Workbooks.Open(path)
                try:
                    wb.SaveAs(path + 'x',FileFormat=51)
                    wb.close()
                except Exception as e:
                    print(e)
            df_wb = pd.read_excel(path + 'x', skiprows=1)
            df_wb = df_wb.dropna(subset=['城市'])
            if "废物处理率" in f:
                df_wb_1 = df_wb.iloc[:, :4]
                df_wb_2 = df_wb.iloc[:, 4:]
                df_wb_1.columns = self.three_cols
                df_wb_2.columns = self.three_cols
                df_wb = pd.concat([df_wb_1, df_wb_2])
                df_wb = df_wb.dropna(subset=['city'])    
            else:
                df_wb.columns = self.five_cols
            if df_m is None:
                df_m = df_wb
            else:
                df_m = pd.merge(df_m, df_wb)
        df_m['year'] = 2007
        df_m.drop_duplicates(subset='city', inplace=True)
        return df_m
    
    def m_2008(self,root_dir):
        """
        2008
        """
        files = files = glob.glob(root_dir + '/*.xlsx')
        df_1 = pd.DataFrame({})
        df_2 = pd.DataFrame({})
        for i in files:
            if "废物处理率" in i:
                df_wb = pd.read_excel(i, skiprows=2)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_1 = pd.concat([df_1, df_wb])
            else:
                df_wb = pd.read_excel(i, skiprows=1)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_2 = pd.concat([df_2, df_wb])
        df_1.columns = self.three_cols
        print(df_1.shape, df_2.shape)
        df_2.columns = self.five_cols
        df_wb = pd.merge(df_1, df_2)
        df_wb['year'] = 2008
        df_wb.drop_duplicates(subset='city', inplace=True)
        return df_wb
    
    def m_2009(self, path):
        """
        表：'2-41 环境治理主要指标(全市)', '2-42 废物处理率(全市)'
        """
        df_m = None
        for i in ['2-41 环境治理主要指标(全市)', '2-42 废物处理率(全市)']:
            if '环境治理' in i:
                df_wb = pd.read_excel(path, sheet_name=i, skiprows=6)
                df_wb.columns = self.five_cols
            else:
                df_wb = pd.read_excel(path, sheet_name=i, skiprows=6)
                df_wb.columns = self.three_cols
            if df_m is None:
                df_m = df_wb
            else:
                df_m = pd.merge(df_m, df_wb)
        df_m['year'] = 2009
        df_m.drop_duplicates(subset='city', inplace=True)
        return df_m
    
    def m_2010(self, path):
        """
        表: '2-41', '2-42'
        """
        df_m = None
        for i in ['2-41', '2-42']:
            if i == '2-41':
                df_wb = pd.read_excel(path, sheet_name=i, skiprows=5)
                df_wb.columns = self.five_cols
            else:
                df_wb = pd.read_excel(path, sheet_name=i, skiprows=4)
                df_wb.columns = self.three_cols
            if df_m is None:
                df_m = df_wb
            else:
                df_m = pd.merge(df_m, df_wb)
        df_m['year'] = 2010
        df_m.drop_duplicates(subset='city', inplace=True)
        return df_m
    
    def m_2011(self, root_dir):
        files = glob.glob(root_dir + '/*.xlsx')
        df_1 = pd.DataFrame({})
        df_2 = pd.DataFrame({})
        df_3 = pd.DataFrame({})
        for i in files:
            if "废水" in i:
                df_wb = pd.read_excel(i, skiprows=1)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_1 = pd.concat([df_1, df_wb])
            elif '固体' in i:
                df_wb = pd.read_excel(i, skiprows=2)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_2 = pd.concat([df_2, df_wb])
            else:
                if '2012' in i:
                    skip = 2
                else:
                    skip=1
                df_wb = pd.read_excel(i, skiprows=skip)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_3 = pd.concat([df_3, df_wb])
        df_wb = pd.merge(df_1, df_2).merge(df_3)
        df_wb = df_wb.rename(columns={
            '工业烟(粉)尘去除量(吨)': '工业烟尘去除量(吨)', 
            '工业烟(粉)尘排放量(吨)': '工业烟尘排放量(吨)',
            '污水集中处理率': '城镇生活污水集中处理率',
            '工业二氧化硫产生量(吨)': '工业二氧化硫去除量(吨)',
            '城市': 'city'
        })
        df_wb = df_wb[['city', '工业固体废物综合利用率','城镇生活污水集中处理率','生活垃圾无害化处理率', 
                         '工业废水排放量(万吨)','工业二氧化硫去除量(吨)',
                         '工业二氧化硫排放量(吨)','工业烟尘去除量(吨)','工业烟尘排放量(吨)']]
        df_wb['year'] = 2011
        df_wb.drop_duplicates(subset='city', inplace=True)
        return df_wb
    
    def m_2012(self, root_dir):
        files = glob.glob(root_dir + '/*.xlsx')
        df_1 = pd.DataFrame({})
        df_2 = pd.DataFrame({})
        df_3 = pd.DataFrame({})
        for i in files:
            if "废水" in i:
                df_wb = pd.read_excel(i, skiprows=1)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_1 = pd.concat([df_1, df_wb])
            elif '固体' in i:
                df_wb = pd.read_excel(i, skiprows=2)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_2 = pd.concat([df_2, df_wb])
            else:
                if '2012' in i:
                    skip = 2
                else:
                    skip=1
                df_wb = pd.read_excel(i, skiprows=skip)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_3 = pd.concat([df_3, df_wb])
        df_wb = pd.merge(df_1, df_2).merge(df_3)
        df_wb = df_wb.rename(columns={
            '工业烟(粉)尘去除量(吨)': '工业烟尘去除量(吨)', 
            '工业烟(粉)尘排放量(吨)': '工业烟尘排放量(吨)',
            '污水处理厂集中处理率': '城镇生活污水集中处理率',
            '工业二氧化硫产生量(吨)': '工业二氧化硫去除量(吨)',
            '一般工业固体废物综合利用率': '工业固体废物综合利用率',
            '城市': 'city'
        })
        df_wb = df_wb[['city', '工业固体废物综合利用率','城镇生活污水集中处理率','生活垃圾无害化处理率', 
                         '工业废水排放量(万吨)','工业二氧化硫去除量(吨)',
                         '工业二氧化硫排放量(吨)','工业烟尘去除量(吨)','工业烟尘排放量(吨)']]
        df_wb['year'] = 2012
        df_wb.drop_duplicates(subset='city', inplace=True)
        return df_wb
    
    def m_2013(self, root_dir):
        import copy
        files = glob.glob(root_dir + '/*.xls')
        df_1 = None
        df_2 = None
        df_3 = None
        for i in files:
            if "废水" in i:
                df_1 = pd.read_excel(i, skiprows=7)
                df_1 = df_1.dropna(subset=['City'])
                df_1 = df_1.query("城市!='城 市'")
            elif '固体' in i:
                df_2 = pd.read_excel(i, skiprows=6)
                df_2 = df_2.dropna(subset=['City'])
                df_2 = df_2.query("城市!='城 市'")
            else:
                df_3 = pd.read_excel(i, skiprows=6)
                df_3 = df_3.dropna(subset=['City'])
                df_3 = df_3.query("城市!='城 市'")
        df_1_c = list(df_1['城市'])
        df_2_c = list(df_2['城市'])
        df_3_c = list(df_3['城市'])
        for i in copy.deepcopy(df_1_c):
            if (i in df_2_c) and (i in df_3_c):
                df_1_c.remove(i)
                df_2_c.remove(i)
                df_3_c.remove(i)
        #print(df_1_c, df_2_c, df_3_c)
        replace = {
            '浙江省 杭州市': '杭州市',
            '安徽省 合肥市': '合肥市',
            '宪湖市': '芜湖市',
            '福建省 福州市': '福州市',
            '甫田市': '莆田市',
            '江西省 南昌市': '南昌市',
            '荷泽市': '菏泽市',
            '召_市': '邵阳市'}  
        df_1 = df_1.replace(replace)
        df_2 = df_2.replace(replace)
        df_3 = df_3.replace(replace)
        df_wb = pd.merge(df_1, df_2).merge(df_3)
        df_wb.rename(columns={'城市': 'city'}, inplace=True)
        df_wb = df_wb[['city', '工业固体废物综合利用率','城镇生活污水集中处理率','生活垃圾无害化处理率', 
                         '工业废水排放量(万吨)','工业二氧化硫去除量(吨)',
                         '工业二氧化硫排放量(吨)','工业烟尘去除量(吨)','工业烟尘排放量(吨)']]
        df_wb['year'] = 2013
        df_wb.drop_duplicates(subset='city', inplace=True)
        return df_wb
    
    def m_2014(self, root_dir):
        files = glob.glob(root_dir + '/*.xlsx')
        df_1 = pd.DataFrame({})
        df_2 = pd.DataFrame({})
        df_3 = pd.DataFrame({})
        for i in files:
            if "废水" in i:
                df_wb = pd.read_excel(i, skiprows=1)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_1 = pd.concat([df_1, df_wb])
            elif '固体' in i:
                df_wb = pd.read_excel(i, skiprows=2)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_2 = pd.concat([df_2, df_wb])
            else:
                df_wb = pd.read_excel(i, skiprows=2)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_3 = pd.concat([df_3, df_wb])
        df_wb = pd.merge(df_1, df_2).merge(df_3)
        df_wb = df_wb.rename(columns={
            '工业烟(粉)尘去除量': '工业烟尘去除量(吨)', 
            '工业烟(粉)尘排放量': '工业烟尘排放量(吨)',
            '污水处理厂集中处理率': '城镇生活污水集中处理率',
            '工业二氧化硫产生量(吨)': '工业二氧化硫去除量(吨)',
            '城市': 'city',
            '一般工业固体废物综合利用率': '工业固体废物综合利用率',
            
        })
        df_wb = df_wb[['city', '工业固体废物综合利用率','城镇生活污水集中处理率','生活垃圾无害化处理率', 
                         '工业废水排放量(万吨)','工业二氧化硫去除量(吨)',
                         '工业二氧化硫排放量(吨)','工业烟尘去除量(吨)','工业烟尘排放量(吨)']]
        df_wb['year'] = 2014
        df_wb.drop_duplicates(subset='city', inplace=True)
        return df_wb
    
    def m_2015(self, path):
        df_wb = pd.read_excel(path, skiprows=5).query("`城 市`==`城 市`").set_index('城 市').iloc[:, -8:].reset_index()
        columns = ['city','工业废水排放量(万吨)', '工业二氧化硫去除量(吨)', '工业二氧化硫排放量(吨)', '工业烟尘去除量(吨)', 
                   '工业烟尘排放量(吨)', '工业固体废物综合利用率','城镇生活污水集中处理率','生活垃圾无害化处理率', ]
        df_wb.columns = columns
        df_wb['year'] =2015
        df_wb.drop_duplicates(subset='city', inplace=True)
        return df_wb
    
    def m_2016(self, path):
        df_wb = pd.read_excel(path, sheet_name=['废水', '烟尘', '废弃物'])
        df_m = pd.merge(df_wb['废水'].drop(columns='City'), df_wb['烟尘'].drop(columns='City')).merge(df_wb['废弃物'].drop(columns='City'))
        df_m.rename(columns={'城市': 'city'}, inplace=True)
        columns = ['city','工业废水排放量(万吨)', '工业二氧化硫去除量(吨)', '工业二氧化硫排放量(吨)', '工业烟尘去除量(吨)', 
                   '工业烟尘排放量(吨)', '工业固体废物综合利用率','城镇生活污水集中处理率','生活垃圾无害化处理率', ]
        df_m = df_m[columns]
        df_m['year'] = 2016
        df_m.drop_duplicates(subset='city', inplace=True)
        return df_m
    
    def m_2017(self, path):
        df_wb = pd.read_excel(path, sheet_name=['废水', '烟尘', '处理率'])
        df_m = pd.merge(df_wb['废水'].drop(columns='City'), df_wb['烟尘'].drop(columns='City')).merge(df_wb['处理率'].drop(columns='City'))
        df_m.rename(columns={'城市': 'city'}, inplace=True)
        columns = ['city','工业废水排放量(万吨)', '工业二氧化硫排放量(吨)', 
                   '工业烟尘排放量(吨)', '工业固体废物综合利用率','城镇生活污水集中处理率','生活垃圾无害化处理率', ]
        df_m = df_m[columns]
        df_m['year'] = 2017
        df_m.drop_duplicates(subset='city', inplace=True)
        return df_m
    
    def m_2018(self, path):
        df_wb = pd.read_excel(path, sheet_name='环境相关').rename(columns={'城市': 'city'})
        df_wb['year'] = 2018
        return df_wb
    
    def m_2019(self, root_dir):
        files = glob.glob(root_dir + "/*.xls")
        df_1 = pd.DataFrame({})
        df_2 = pd.DataFrame({})
        df_3 = pd.DataFrame({})
        for i in files:
            if "废水" in i:
                df_wb = pd.read_excel(i, skiprows=1)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_1 = pd.concat([df_1, df_wb])
            elif '固体' in i:
                df_wb = pd.read_excel(i, skiprows=1)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_2 = pd.concat([df_2, df_wb])
            else:
                df_wb = pd.read_excel(i, skiprows=1)
                if '摘编' in df_wb.iloc[-1, 0]:
                    df_wb = df_wb.iloc[:-1,:]
                df_3 = pd.concat([df_3, df_wb])
        df_1.dropna(subset=['城市'], inplace=True)
        df_2.dropna(subset=['城市'], inplace=True)
        df_3.dropna(subset=['城市'], inplace=True)
        df_m = pd.merge(df_1, df_2, on='城市').merge(df_3, on='城市')
        df_m.columns = [i.strip() for i in df_m.columns]
        df_m.rename(columns={
            '一般工业固体废物综合利用率(%)':'工业固体废物综合利用率',
            '污水处理厂集中处理率(%)': '城镇生活污水集中处理率',
            '生活垃圾无害化处理率(%)': '生活垃圾无害化处理率',
            '工业烟(粉)尘排放量(吨)': '工业烟尘排放量(吨)',
            '城市': 'city'
            
        }, inplace=True)
        df_m = df_m[['city', '工业固体废物综合利用率','城镇生活污水集中处理率','生活垃圾无害化处理率', 
                    '工业废水排放量(万吨)', '工业二氧化硫排放量(吨)','工业烟尘排放量(吨)']]
        for i in df_m.drop(columns='city').columns:
            df_m[i] = df_m[i].map(lambda x: x.strip() if isinstance(x, str) else x)
            df_m[i] = df_m[i].map(lambda x: x if x else None)
            try:
                df_m[i] = df_m[i].astype(float)
            except Exception as e:
                for j in df_m[i]:
                    try:
                        float(j)
                    except Exception as e:
                        print(j)
                        raise TypeError
        
        df_m['year'] = 2019
        df_m.drop_duplicates(subset='city', inplace=True)
        return df_m
    
    def year_book_full(self, path_1, path_2):
        """
        path_1: data/统计年鉴/年鉴单位不统一指标.xlsx
        path_2: data/统计年鉴/1996-2020全国数据汇总_2019数据不准需要用最新2019正式发布.xlsx
        """
        df_units = pd.read_excel(path_1)
        df_full = pd.read_excel(path_2, sheet_name='Sheet1')

        df_full = df_full.query("指标!='指标'")
        df_full = pd.merge(df_full, df_units, on=['指标', '单位'], how='left')
        df_full['weight'] = df_full['weight'].fillna(1)
        for i in range(1995, 2019):
            df_full[f'{i}'] = df_full[f'{i}'] * df_full[f'weight']
        df_full['单位'] = df_full[['单位', 'units']].apply(lambda x: x['单位'] if pd.isnull(x['units']) else x['units'], axis=1)
        df_full.drop(columns=['weight', 'units'], inplace=True)
        df_full['指标'] = df_full['指标'] + '-' + df_full['单位']
        df_full.drop(columns=['单位', '频度', '2019', '2020'], inplace=True)
        df_full = df_full.set_index(['指标', '地区'])
        df_full = df_full.stack().reset_index()
        df_full.columns = ['指标', 'city', 'year', 'value']
        df_full = df_full.groupby(['city', 'year', '指标']).mean().unstack().reset_index()
        columns = [i[0] if i[0] != 'value' else i[1] for i in df_full.columns]
        df_full.columns = columns
        return df_full

        
def gen_yearbook_data(root_dir, full_yb_path, not_unique_index_path):
    cr = ChinaRegion()
    df_all = pd.DataFrame({})
    for i in os.listdir(root_dir):
        print(i)
        path = root_dir + f'/{i}'
        try:
            year = re.search(r".*?(\d+).*?", path)[1]
            df_temp = eval(f"YearBook().m_{year}('{path}')")
            df_all = pd.concat([df_all, df_temp])
        except Exception as e:
            print(e, i)
    df_all['city'] = df_all['city'].map(lambda x: x.strip())
    city_fix = {
        '毫州市': '毫州',
        '尤锡市': '无锡',
        '秦％岛市': '秦皇岛',
        '折州市': '忻州',
        '抚顾市': '抚顺',
        '楼坊市': '廊坊',
        '乌兰察市市': '乌兰察布',
        '掄林市': '榆林',
        '荷泽市': '菏泽',
        '常徳市': '常德',
    }
    df_all['city'] = df_all['city'].replace(city_fix)
    df_full = YearBook().year_book_full(not_unique_index_path, full_yb_path)
    df_full['year'] = df_full['year'].astype(int)
    
    df_city = pd.read_csv("code/city_simple.csv", encoding='gbk')
    df_all['city'] = df_all['city'].map(lambda x: cr.simple_location(x))
    
    df_merge_all = pd.merge(df_all, df_full, on=['city', 'year'])
    
    # 转换成面板
    all_city = list(set(df_merge_all['city']))
    max_year, min_year = df_merge_all['year'].max(), df_merge_all['year'].min()
    time_span = (FlexibleTime(max_year) - FlexibleTime(min_year)).get_range()
    index_list = []
    for i in all_city:
        for j in time_span:
            index_list.append([i, j])
    df_index = pd.DataFrame(index_list, columns=['city', 'year'])
    df_merge_all['year'] = df_merge_all['year'].map(lambda x: str(int(x)) if not pd.isnull(x) else None)
    df_merge_all = pd.merge(df_index, df_merge_all, how='left')
    
    return df_merge_all


class CountyBook:
    
    def __init__(self, file_root_path, save_path):
        """
        县级统计年鉴
        仅支持分省份分年度单独文件的县域统计年鉴，目录结构应为：
        
        |根目录
        |---县级*年鉴*数据
        |------分省份.xls
        """
        self.replace_item = {
            '各种社会福利收养性单位床位数': '社会福利院床位数',
            '各种社会福利收养性单位数': '社会福利院数',
            '固定电话用户': '本地电话年末用户',
            '粮食产量': '粮食总产量',
            '居民储蓄存款余额': '城乡居民储蓄存款余额',
            '农业增加值': '第一产业增加值',
            '固定资产投资(不含农户)': '城镇固定资产投资完成额',
            '行政区域面积': '行政区域土地面积',
            '公共财政收入': '地方财政一般预算收入',
            '公共财政支出': '地方财政一般预算支出',
            '医院卫生院床位数': '医疗卫生机构床位数',
            '规模以上工业总产值(现价)': '规模以上工业总产值',
            '规模以上工业企业单位数': '规模以上工业企业个数',
            '规模以上丁业企业单位数': '规模以上工业企业个数',
            '各种社会福利收养件单位床位数': '社会福利院床位数',
            '一般公共预算收入': '地方财政一般预算收入',
            '·般公共预算收入': '地方财政一般预算收入',
            '一般公共预算支出': '地方财政一般预算支出',
            '设施农业占地(水面)面积': '设施农业占地面积',
            
        }
        self.replace_index = {
            '普通中学在校学生数’': '普通中学在校学生数',
            '年末金融机构各项贷款余额．': '年末金融机构各项贷款余额',
            '本地电话年未用户': '本地电话年末用户',
            '其中；农林牧渔业': '其中：农林牧渔业',
            '．各种社会福利收养性单位床位数': '各种社会福利收养性单位床位数',
            '其中t乡村户数': '其中：乡村户数',
            '乡(镇)个数．': '乡(镇)个数',
            '棉花产最': '棉花产量',
            '规模以上1:业企业单位数': '规模以上工业企业单位数',
            '户籍入口': '户籍人口',
            '户籍人门': '户籍人口',
            '固定电话用户"': '固定电话用户',
            '第·产业增加值': '第一产业增加值',
            '·般公共预算支出': '一般公共预算支出',
            '户籍人n': '户籍人口',
            '·般公共预算收入': '一般公共预算收入',
            '医疗卫生机构庆位数': '医疗卫生机构床位数',
            '户籍人u': '户籍人口',
            '各种社会福利收养性单位床位数"': '各种社会福利收养性单位床位数',
            '设施农业占(水面)面积': '设施农业占地(水面)面积',
            '没施农业占地(水面)面积': '设施农业占地(水面)面积',
            '油料产最': '油料产量',
            '公共财政收人': '公共财政收入',
            '棉化产量': '棉花产量',
            '公共财政收': '公共财政收入',
            '‘第三产业从业人员': '第三产业从业人员',
            '第：产业从业人员': '第二产业从业人员',
            '.牧业增加值': '牧业增加值',
            '各项税收入': '各项税收',
            '中等职业教育学校在校学牛数': '中等职业教育学校在校学生数',
            '固定电话月户': '固定电话用户',
            '棉花产最': '棉花产量',
            '居民储蒂存款余额': '居民储蓄存款余额',
            '油料产景': '油料产量',
            '规模以上丁业总产值': '规模以上工业总产值',
            '地区生产总值值': '地区生产总值',
            '地区牛产总值': '地区生产总值',
            '肉类总产最': '肉类总产量',
            '规模以工业企业单位数': '规模以上工业企业单位数',
            '固定电活用户': '固定电话用户',
            '机收而积': '机收面积',
            '牧业增加值值': '牧业增加值',
            '同定资产投资': '固定资产投资',
            '规模以上业企业单位数': '规模以上工业企业单位数',
            '同定电话用户': '固定电话用户',
            '年末金融机构各项货款余额': '年末金融机构各项贷款余额',
            '农Ik机械总动力': '农业机械总动力',
            '各种会福利收养性单位床位数': '各种社会福利收养性单位床位数',
            '棉花产量_': '棉花产量',
            '第二产业从业入员': '第二产业从业人员',
            '第一产业增加值值': '第一产业增加值',
            '第三产业从业入员': '第三产业从业人员',
            '各种社会福利收养件单位床位数': '各种社会福利收养性单位床位数',
            '规模以上丁业企业单位数': '规模以上工业企业单位数',
            '第三产业业从业人员': '第三产业从业人员',
            '行政区域面积"': '行政区域面积',
            '地方财政一般预算收人': '地方财政一般预算收入',
            '社会福利收养性单位数': '各种社会福利收养性单位数',
            '同定资产投资(不含农户)': '固定资产投资(不含农户)',
            '也方财政一般预算收入': '地方财政一般预算收入',
            '也方财政一般预算支出': '地方财政一般预算支出',
            '曲料产量': '油料产量',
            '第1产业增加值': '第一产业增加值',
            '第了产业增加值': '第一产业增加值',
            '年未总户数': '年末总户数',
            '其中:农林牧渔业': '其中：农林牧渔业',
            '年未总人口': '年末总人口',
            '年术总户数': '年末总户数',
            '其中；乡村户数': '其中：乡村户数',
            '年末总人口1': '年末总人口',
            '城镇同定资产投资完成额': '城镇固定资产投资完成额',
            '年未单位从业人员数': '年末单位从业人员数',
            '第-产业增加值': '第一产业增加值',
            '第：产业增加值': '第二产业增加值',
            '医疗卫牛机构床位数': '医疗卫生机构床位数',
            '年未金融机构各项贷款余额': '年末金融机构各项贷款余额',
            '固定定电话用户': '固定电话用户',
            '行政区域而积': '行政区域面积',
             
        }
        self.replace_unit = {
            '卢': '户',
            '爪': '个',
            '由': '户',
            '小': '个',
            '记': '户',
            '白': '户',
            '方人': '万人',
            '万千': '万千瓦特',
            '平万公里': '平方公里',
            '入': '人',
            '万入': '万人',
            '力元': '万元',
            '万万元': '万元',
            '.万元': '万元',
            '万下瓦特': '万千瓦特',
            '.人': '人',
            '／万元': '万元',
            '万兀': '万元', 
            'P': '户',
            '.户': '户',
            '万f瓦特': '万千瓦特',
            '刀元': '万元',
            '刀千瓦特': '万千瓦特',
            '方元': '万元',
        }
        self.root_path = file_root_path
        self.save_path = save_path
            
    def get_one_province(self, data):
        locations = []
        columns = list(data.columns)
        for i in columns:
            data[i] = data[i].map(lambda x: x.strip().replace(' ', '').replace('、', '') if isinstance(x, str) else x)
        for m,i in enumerate(columns[:-1]):
            for n,j in enumerate(data[i]):
                if j == '指标' and data.iloc[n, m + 1] == '单位':
                    locations.append([n, m])
        for m, i in enumerate(locations):
            na_flag = False
            for n, j in enumerate(data.iloc[i[0], i[1]:]):
                if pd.isnull(j):
                    if (not pd.isnull(data.iloc[i[0] - 1, i[1] + n])) and (not pd.isnull(data.iloc[i[0] + 1, i[1] + n])):
                        data.iloc[i[0], i[1] + n] = data.iloc[i[0] - 1, i[1] + n] + data.iloc[i[0] + 1, i[1] + n]
                        na_flag = True
                        if data.shape[1] == i[1] + n + 1:
                            locations[m].append(i[1] + n + 1)
                            break
                    else:
                        locations[m].append(i[1] + n)
                        break
                elif data.shape[1] == i[1] + n + 1:
                    locations[m].append(i[1] + n + 1)
                    break
                else:
                    pass
            if na_flag:
                #print(m, locations[m])
                #print(data.iloc[i[0], i[1]: locations[m][2]], m)
                data.iloc[i[0] + 1, i[1]: locations[m][2]] = data.iloc[i[0], i[1]: locations[m][2]]
                data.iloc[i[0], i[1]: locations[m][2]] = None 
                locations[m][0] = i[0] + 1
        for m, i in enumerate(locations):
            for n, j in enumerate(data.iloc[i[0]:, i[1]]):
                if pd.isnull(j):
                    locations[m].append(i[0] + n)
                    break
                elif data.shape[0] == i[0] + n + 1:
                    locations[m].append(i[0] + n + 1)
                    break
                else:
                    pass
        #print('sss', locations)
        df_all = None
        for i in locations:
            df_loc = data.iloc[i[0]:i[3], i[1]:i[2]].reset_index().drop(columns='index')
            df_loc[list(df_loc.columns)[0]] = df_loc[list(df_loc.columns)[0]].replace(self.replace_index)
            df_loc[list(df_loc.columns)[1]] = df_loc[list(df_loc.columns)[1]].replace(self.replace_unit)
            if df_all is None:
                df_all = df_loc.rename(columns={list(df_loc.columns)[0]: 'name', list(df_loc.columns)[1]: 'unit'})
            else:
                df_loc = df_loc.rename(columns={list(df_loc.columns)[0]: 'name', list(df_loc.columns)[1]: 'unit'})
                df_all = df_all.merge(df_loc, on=['name', 'unit'], how='outer')
            #print(df_all.shape)
        df_all.columns = [i.replace('\r\n', '') for i in df_all.iloc[0, :]]
        df_all.drop(index=0, inplace=True)
        #print(f"共{len(locations)}张表")
        
        return df_all
    
    def convert_float(self,x):
        try:
            x = float(x)
        except Exception as e:
            x = None
        return x
    
    def decrypt_xls(self, target_path="解除文件保护数据"):
        """解决文件加密保护问题，只针对默认密码有效"""
        contents = glob.glob(self.file_root_path + '/' + "县级*年鉴*数据")
        for i in contents:
            files = glob.glob(i + "/*xls")
            if not os.path.exists(f"{target_path}/{i}"):
                os.mkdir(f"{target_path}/{i}")
            for j in files:
                with open(j, 'rb') as f:
                    workbook = msoffcrypto.OfficeFile(f)
                    workbook.load_key(password="VelvetSweatshop")
                    province_name = j.split('\\')[-1]
                    workbook.decrypt(open(f"{target_path}/{i}/{province_name}", 'wb'))
    
    def gen_yearbook(self):
        contents = glob.glob(self.root_path + "/县级*年鉴*数据")
        for cont in  contents:
            files = glob.glob(cont + "/*.xls")
            df_files = None
            #print(files)
            for file in files:
                #print(file)
                try:
                    df_temp = pd.read_excel(file.replace(r'\\', '/'))
                except Exception as e:
                    try:
                        df_temp = pd.read_excel(file+"x")
                        print("未成功读取文件, 已读取xlsx文件", file)
                    except FileNotFoundError:
                        print("未成功读取文件, 不存在xlsx文件，尝试手动转换成xlsx文件", file)
                        continue
                df_file = self.get_one_province(df_temp).drop_duplicates(subset=['指标', '单位'])
                if df_files is None:
                    df_files = df_file
                df_files = pd.merge(df_files, df_file, on=['指标', '单位'], how='outer')
            year = '20' + re.search(r".*?年鉴(\d+)数据$", cont)[1]
            df_files['year'] = year
            df_files = df_files.filter(regex="^\w+[^_xy]$").query("单位==单位")
            df_files.to_csv(self.save_path + f"/{year}.csv", index=False)
            print(f"{cont}", df_files.shape)
    
    def merge_year(self, save_path):
        files = glob.glob(self.save_path+"/*.csv")
        
        df_all = pd.DataFrame({})
        all_set = set({})
        for n, file in enumerate(files):
            df_file = pd.read_csv(file)
            print(df_file.shape)
            df_file['指标'] = df_file['指标'].replace(self.replace_item)
            file_set = set(df_file['指标'])
            new_set = file_set - all_set
            exclude_set = all_set - file_set
            if n != 0 and len(new_set) != 0:
                print(f"发现新指标{file}:{new_set}")
            if n != 0 and len(exclude_set) != 0:
                print(f"发现缺失指标{file}:{exclude_set}")
            all_set = all_set.union(file_set)
            df_all = pd.concat([df_all, df_file])
        df_all_copy = df_all.copy() # 懒得修改了
        df_all_copy['指标'] = df_all_copy['指标'] + '_' + df_all_copy['单位'] + '-' + df_all_copy['year'].map(lambda x: str(x))
        df_all_copy.drop(columns=['单位', 'year'], inplace=True)
        df_all_copy = df_all_copy.set_index('指标')
        df_all_copy = df_all_copy.stack().reset_index()
        df_all_copy.columns = ['指标', 'county', 'value']
        df_all_copy['year'] = df_all_copy['指标'].map(lambda x: x.split('-')[-1])
        df_all_copy['indexes'] = df_all_copy['指标'].map(lambda x: x.split('-')[0])
        df_all_copy.drop(columns=['指标'], inplace=True)
        df_all_copy['value'] = df_all_copy['value'].map(self.convert_float)
        df_all_copy = df_all_copy.groupby(by=['county', 'year', 'indexes'])['value'].mean().unstack().reset_index()
        df_all_copy['户籍人口_万人'] = df_all_copy[['户籍人口_万人', '户籍人口_人']].apply(lambda x: x['户籍人口_人'] / 10000 if not pd.isnull(x['户籍人口_人']) else x['户籍人口_万人'], axis=1)
        df_all_copy.to_excel(save_path, index=False)
        print("merge_year: 所有年份合并", df_all_copy.shape)
        return df_all_copy

if __name__ == '__main__':
    #df_env = gen_yearbook_data("data/统计年鉴")
    #df_full = YearBook().year_book_full("../data/统计年鉴/年鉴单位不统一指标.xlsx", "../data/统计年鉴/1996-2020全国数据汇总_2019数据不准需要用最新2019正式发布.xlsx")
    df_yb = gen_yearbook_data("data/统计年鉴", "../data/统计年鉴/1996-2020全国数据汇总_2019数据不准需要用最新2019正式发布.xlsx", "../data/统计年鉴/年鉴单位不统一指标.xlsx")