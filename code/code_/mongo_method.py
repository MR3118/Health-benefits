import time
import pymongo
import pandas as pd
from pymongo.errors import DuplicateKeyError
import re
from datetime import datetime
import json

REMOTE_MONGO_HOST = 'mongodb://scrapy:a7621546@127.0.0.1:27017/'
client = pymongo.MongoClient(REMOTE_MONGO_HOST)


class DumpMongo:

    def __init__(self, old_db_name, old_col_name, new_db_name='test', new_col_name='test', info_freq=None):
        """使用说明
        old_db_name: 原始数据库
        old_col_name：原始表名称
        new_db_name：要创建的新表所在数据库
        new_col_name：原始表名称
        info_freq：返回进度的频率，秒
        """
        self.old_col = client[old_db_name][old_col_name]
        self.new_col = client[new_db_name][new_col_name]
        self.old_db_name = old_db_name
        self.new_db_name = new_db_name
        self.info_freq = info_freq

    def update(self, filters, update_keys, type_):
        if type_ == 'set':
            self.old_col.update_many(filters, {'$set': update_keys})
        elif type_ == 'unset':
            self.old_col.update_many(filters, {'$unset': update_keys})  # update_keys={'name': ""}
        else:
            raise KeyError('没有此项功能')

    def add_new(self, old_filter_cond, new_keys, old_use_keys=None, map_func=None, limit=0):
        """
        数据转存到新表
        @:param old_filter_cond：原表查询条件，dict
        @:param new_keys：新增键值，dict，可以是空字典
        @:param old_use_keys：原表键值过滤：{key: 0/1}
        @:param map_func: 对原表数据处理的回调函数
        @:param return: None
        如果新库和新表均和原库原表相同，则为更新原表模式
        """
        if (self.old_col == self.new_col) and (self.old_db_name == self.new_db_name):
            print('目前模式为更新原表模式')
        if old_use_keys is not None:
            data_used = self.old_col.find(old_filter_cond, old_use_keys).limit(limit)
        else:
            data_used = self.old_col.find(old_filter_cond).limit(limit)
        time_all_1 = time.perf_counter()
        time_1 = time.perf_counter()
        counter = 0
        for num, d in enumerate(data_used):
            old_d = dict(d)
            for i in old_d.keys():
                new_keys[i] = old_d[i]
            if map_func is not None:
                new_keys = map_func(new_keys)
            else:
                new_keys = new_keys
            try:
                self.new_col.insert_one(new_keys)
            except DuplicateKeyError:
                #if (self.old_col == self.new_col) and (self.old_db_name == self.new_db_name):
                #self.new_col.update_one({'_id': new_keys['_id']}, {'$set': new_keys})
                #else:
                pass

            if self.info_freq is not None:
                if num % self.info_freq == 0 and (num != 0):
                    time_2 = time.perf_counter()
                    speed = self.info_freq / (time_2 - time_1)
                    print('已导入新集合{}条数据，当前速度：{}条/秒'.format(num + 1, round(speed, 0)))
                    time_1 = time.perf_counter()
            counter = num

        time_all_2 = time.perf_counter()
        time_all = time_all_2 - time_all_1
        hours = time_all // 3600
        mins = (time_all % 3600) // 60
        sec = (time_all % 3600) % 60
        print('导出完毕, 共处理{}条数据, 用时{}h{}m{}s'.format(counter, hours, mins, round(sec, 0)))

    def download(self, cond, save_path, chunk_size, use_keys=None, map_func=None, limit=0, rename=None, order=None, skip=0):
        """
        :param limit: 提取文档数，默认0，即不限制
        :param cond: 过滤条件
        :param save_path: 所要保存的文件名称和位置
        :param chunk_size: 每多少条数据存储一次, 该值必须小于数据总量
        :param use_keys: 筛选特定的key
        :param map_func: 传入函数，对提取的每一条数据进行处理
        :param order: 输出数据列名顺序
        :param rename: 改变列的名字
        :return: None
        """
        if use_keys is not None:
            data_used = self.old_col.find(cond, use_keys).limit(limit).skip(skip)
        else:
            data_used = self.old_col.find(cond).limit(limit).skip(skip)
        res = []
        time_all_1 = time.perf_counter()
        time_1 = time.perf_counter()
        num = 0
        for i, j in enumerate(data_used):
            if map_func:
                j = map_func(j)
                if j:
                    res.append(j)
                else:
                    pass
            else:
                res.append(j)
            if i % chunk_size == 0:
                print(i)
                dfs = pd.DataFrame(res)
                if rename:
                    dfs.rename(columns=rename, inplace=True)
                if order:
                    dfs = dfs[order]
                if i == 0:
                    dfs.to_csv(save_path, index=False, encoding='utf8', sep='\t')
                else:
                    dfs.to_csv(save_path, index=False, encoding='utf8', mode='a+', header=None, sep='\t')
                time_2 = time.perf_counter()
                speed = chunk_size / (time_2 - time_1)
                print('已导出{}条数据，当前速度：{}条/秒'.format(i + 1, round(speed, 0)))
                time_1 = time.perf_counter()
                res = []
            num += 1
        dfs_res = pd.DataFrame(res)
        dfs_res.to_csv(save_path, index=False, encoding='utf8', mode='a+', header=None, sep='\t')
        time_all_2 = time.perf_counter()
        time_all = time_all_2 - time_all_1
        hours = time_all // 3600
        mins = (time_all % 3600) // 60
        sec = round((time_all % 3600) % 60, 0)
        print('导出完毕, 共导出{}条数据, 用时{}小时{}分{}秒'.format(num, hours, mins, sec))

    @staticmethod
    def upload(data_path, db_name, col_name, encoding='utf8', sep=',', header='infer', main_key=None):
        up_data = pd.read_csv(data_path, encoding=encoding, sep=sep, header=header)
        if main_key:
            up_data.rename(columns={main_key: '_id'}, inplace=True)
        col = client[db_name][col_name]
        for i in up_data.to_dict(orient='records'):
            try:
                col.insert_one(i)
            except DuplicateKeyError:
                pass


if __name__ == '__main__':
    dt = datetime.now()
    base_time = datetime(year=2020, month=3, day=22)
    # DumpMongo.upload("需要爬取转发的mid.csv", 'blog', 'blog_env_id', main_key='mid')
    # DumpMongo('pneumonia', 'SearchUserContent').update({'root_mid': {'$ne': None},'get_time': {'$gt': base_time}}, {'is_ori': 0}, type='set')
    # DumpMongo('pneumonia', 'SearchUserContent').update({'root_mid': None, 'get_time': {'$gt': base_time}}, {'is_ori': 1}, type='set')
    # DumpMongo('pneumonia', 'SearchUserContent').update({'get_time': {'$gt': base_time}}, {'has_cmt': 0}, type='set')
    # DumpMongo('pneumonia', 'user_all_id').update({'build_flag': 1}, {'build_flag': 0, 'time': base_time}, type='set')
    def parse_time(x):
        try:
            pb_time = x['publish_time']
        except KeyError:
            return None
        tool = x['tool']
        if pb_time or tool:
            times = pb_time.split(' ')
            try:
                dt = re.sub(r'-', '', times[0])
                hour = int(times[1].split(':')[0])
                x['hour'] = hour
                x['dt'] = dt
            except Exception as e:
                print(x['_id'], pb_time)
                print(e)
                if tool:
                    pb_time = tool
                s = pb_time.split(' ')
                date = re.findall(r'\d+', s[0])
                ss = s[1].split('\xa0')
                time = ss[0]
                tool = ss[1][2:]
                publish_time = '2020-' + '-'.join(date) + ' ' + time
                dt = '2020' + ''.join(date)
                hour = int(time.split(':')[0])
                x['publish_time'] = publish_time
                x['tool'] = tool
                x['hour'] = hour
                x['dt'] = dt
        else:
            with open('E:/刘杰/微博/logs.txt', 'a') as f:
                f.write(x['_id'] + '\n')
            x['hour'] = None
            x['dt'] = None
        return x
    max_num = 36445325
    flag = True
    batch_size = 1000000
    i = 0
    """
    while flag:
        if (i + 1) * batch_size >= max_num:
            flag = False
        DumpMongo('blog', f'blog_env_repost_2').download({}, f'C:/Data/repost_{i}.csv', 500000,limit=batch_size, skip=i*batch_size)
        i += 1
    """
    DumpMongo('blog', f'blog_env_repost_2').download({}, f'C:/Data/blog_env_repos_2.csv', 1000000,limit=0,use_keys={'_id': 1, 'publish_time': 1, 'content': 1, 'user_id': 1, 'root_mid': 1, 'parent_mid': 1},)


    def new_url(x):
        url = x['user_url']
        n_url = re.sub(r'https://weibo\.cn', 'https://weibo.com', url)
        x['user_url'] = n_url
        return x


    def rename(x):
        user_url = x['user_url']
        x.pop('user_url')
        x['_id'] = user_url
        return x

    def add_https(x):
        url = x['_id']
        if re.search(r'^https.*?', url):
            url = url
        else:
            url = 'https://weibo.cn' + url
        x['_id'] = url
        return x


    def format_str(x):
        c = x['content']
        if c:
            x['content'] = ''.join(c.split('\t'))
        else:
            x['content'] = c
        rc = x['root_content']
        if rc:
            x['root_content'] = ''.join(rc.split('\t'))
        else:
            x['root_content'] = rc

        return x

    # 从转发评论中提取用户地址
    # DumpMongo("disaster", 'UserInfoItem', "pneumonia", 'user_all_id_test', 100).add_new({'verify_type': None}, {'build_flag': 0, 'put_time': base_time}, old_use_keys={'_id': 1}, map_func=None, limit=1000)
    # DumpMongo("pneumonia", 'all_verify_user_id_copy', "pneumonia", 'all_verify_user_id', 100000).add_new({},{'build_flag': 1},old_use_keys={'_id': 1}, map_func=None, limit=0)
    # 从评论数据里导出url至url库
    # DumpMongo("pneumonia", 'CommentItem', "disaster", 'user_url', 100000) \
    #    .add_new({'get_time': {'$gte': base_time}}, {'time': datetime.now(), 'build_flag': 'no'}, old_use_keys={'user_url': 1}, map_func=rename)

    # 从新认证用户导入url库
    def uid2id(x):
        x['_id'] = x['user_id']
        del x['user_id']
        return x
    # DumpMongo("blog", 'blog_env_repost', "user_info", 'env_repost_user_id', 100000) \
    #   .add_new({}, {'time': datetime.now(), 'build_flag': 0}, old_use_keys={'user_id': 1}, map_func=uid2id)

    # 更新自己
    # DumpMongo("pneumonia", 'verify_5337', "pneumonia", 'test', 10000)\
    #    .add_new({}, {'error': None, 'verify_info': None},
    #             old_use_keys={'_id': 1, 'user_url': 1}, map_func=new_url)

    def rebuild(x):
        data = json.loads(x['data_full'])
        all_text = []
        x['_id'] = re.search(r"_-_(\d+)$", data['data']['cardlistInfo']['containerid'])[1]
        for i in data['data']['cards']:
            # print(i['card_group'])
            try:
                text = i['card_group'][0]['desc']
                all_text.append(text)
            except KeyError:
                all_text = None
        if all_text:
            final = "T".join(all_text)
        else:
            final = None
        x['data_full'] = final
        return x

    #DumpMongo("pneumonia", 'SearchUserContent', "pneumonia", 'repost_mid', 10000).add_new({'is_ori': 1, 'repost_num': {'$gt': '0'}, 'root_nick_name': None}, {'build_flag': 0, 'time': base_time}, old_use_keys={'_id': 1, 'repost_num': 1}, map_func=trans_id, limit=0)
    #DumpMongo("pneumonia", 'industry_type_and_first_blog_date', "pneumonia", 'suspect_half_year_user', 100000).add_new({'error_dt': 'NoTimeLine', 'error_info': 'NoInfo'},{'build_flag': 0, 'used': 0},old_use_keys={'_id': 1},map_func=None,limit=0)
    cols = ['_id', 'comment_num', 'content', 'dt', 'event_type', 'geo', 'get_time', 'hour', 'keyword', 'like_num',
            'nick_name', 'publish_time', 'repost_num', 'root_comment_num', 'root_content', 'root_like_num', 'root_mid',
            'root_nick_name', 'root_repost_num', 'tool', 'user_id']
    """
    for i in range(11, 14):
        DumpMongo('pneumonia', f'SearchUserContent{i}').download(
            {}, f'F:/数据备份/微博数据/user_content_{i}.csv', 1000000,
            use_keys=None, map_func=parse_time, order=cols, limit=0)
    """
    # DumpMongo("user_info", 'env_repost_user_id').update({'build_flag': 1}, {'build_flag': 0}, 'set')
    # DumpMongo("pneumonia", 'SearchUserContent').upload("C:/Projects/visitor_account_copy1.csv", 'Sina', 'visitor_account', sep=',')
    
    

