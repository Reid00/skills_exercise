import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import chain
import datetime
import re

class Demo():
    """
    this is a demo test for any idea
    """
    def __init__(self, statements=r'this A is a demo for any idea'):
        # super().__init__()
        self.statements = statements

    @classmethod
    def __str__(cls):
        return 'this is a demo test for any idea'

    @staticmethod
    def test_chain():
        lst = [['this', 'is'], [1, 2, 3, 4], ['what', ['is', 'your idea']]]
        lst = list(chain(*lst))
        print(lst)

    def normalize_available_date(self,available_str):
        month_dic = {
            'Dec': 12,
            'Nov': 11,
            'Oct': 10,
            'Sep': 9,
            'Aug': 8,
            'Jul': 7,
            'Jun': 6,
            'May': 5,
            'Apr': 4,
            'Mar': 3,
            'Feb': 2,
            'Jan': 1
        }
        time =available_str.replace('Available:','').strip()
        if time== 'Now' or time == 'now':
            time=datetime.datetime.now()
            return time
        else:
            year=int(datetime.datetime.now().year)
            tmp_month=time.split(' ')[0]
            month= int(month_dic[tmp_month])
            day=int(time.split(' ')[1])
            time=datetime.datetime(year,month,day)
            return time

    def fill_values_by_id(self,path1,path2):
        data1=pd.read_excel(path1,encoding='utf-8-sig')
        data1.drop_duplicates(subset=['歌曲id'],keep='first',inplace=True)
        print(data1.shape)
        print(data1.head())
        data2=pd.read_excel(path2,encoding='utf-8-sig')
        id_counts=data2['歌曲id'].value_counts()
        # 统计歌曲id 出现两次的数量
        print(id_counts[id_counts>1].count())
        #统计30836270 出现的次数
        print(data1.count())
        data2.drop_duplicates(subset=['歌曲id'],keep='first',inplace=True)
        print(data2.shape)
        order_list=data1['歌曲id']
        data2['歌曲id']=data2['歌曲id'].astype('category')
        data2['歌曲id'].cat.set_categories(order_list,inplace=True)
        data2.sort_values(by=['歌曲id'],inplace=True)
        output_name=Path(r'D:\download_D\1230_小万内容清洗\0109\reorder_res.xlsx')
        data2.to_excel(output_name, index=None, header=True, encoding='utf-8')
        print(data2.head())

    def rm_blank(self,path,*columns):
        data=pd.read_excel(path,encoding='utf8-sig')
        print(data.head())
        output_name=Path(r'D:\download_D\1230_小万内容清洗\0109\not_rm_blank.xlsx')
        data.to_excel(output_name,index=None, header=True, encoding='utf8')
        for col in columns:
            data[col]=data[col].apply(lambda x:re.sub(r'\s','',str(x)))
            print(data.head())
        # output_name=Path(r'D:\download_D\1230_小万内容清洗\0109\rm_blank.xlsx')
        # data.to_excel(output_name,index=None, header=True, encoding='utf8')

    def hanming_distance_int(self,int1,int2):
        """
        计算两个整数间的汉明距离
        """
        return bin(x^y).count('1')

    def hanming_distance_str(self,s1,s2):
        """
        计算两个字符串之间的汉明距离
        """
        if len(s1)!=len(s2):
            raise ValueError(r'undefined for sequence of unequal length')
        return sum(el1!=el2 for el1,el2 in zip(s1,s2))

    def groupby_exercise(self):
        data=pd.DataFrame({
            'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]
         })
        print(data.head())
        grouped=data.groupby(['Year'])
        print(grouped.groups)
        print('===='*9)
        for name,group in grouped:
            print(name)
            print(group)
        print('===='*9)
        print(grouped['Points'].agg(np.mean))
        print('===='*9)
        print(grouped.agg(np.mean))
        print(grouped.agg(np.size))
        print('===='*9)
        filtered=grouped.filter(lambda  x: len(x)>=3)
        print(filtered)
        filtered=grouped.filter(lambda  x: len(x)>=5)
        print(filtered)
        print(grouped[['Points']].agg(np.size))
        print(grouped[['Points']].agg(np.min))
        # print(grouped[['Points']].agg(np.count))
        print('===='*9)
        print(grouped[['Points']].agg(np.size).nlargest(2,'Points'))
        print(grouped[['Points']].agg(np.max))
        print('===='*9)
        print(grouped.apply(lambda x: x.nlargest(2,'Points')))
        print(grouped['Points'].apply(lambda x: x.nlargest(2)))
        print(grouped.apply(lambda x: x.nlargest(2,'Points')).sum(level='Year'))

    def gen(self,n):
        print('you talked me.')
        while n>0:
            print('before yield')
            yield n
            n-=1
            print('after yield')

    def lst_test(self):
        l1=set(['a','b'])
        l2=set(['a'])
        if l1 & l2:
            print(max(map(len,l1&l2)))

    def read_first_col(self):
        payload= pd.read_csv(r'C:\Users\v-baoz\Downloads\OutPayloads.tsv',sep='\t',header=None)
        payload.columns=['url','date','content']
        payload['url'].to_csv(r'C:\Users\v-baoz\Downloads\OutUrls.tsv',index=False)

    def pandas_test(self):
        data= pd.DataFrame({'name':['a','b'],'age':[1,2],'gender':[np.nan,np.nan],'other':[np.nan,'u']})
        print(data.isnull())
        print(data.isnull().sum())
        print(data.isnull().any())
        print(data.isnull().all())
        print(data.isnull().all().describe())
        x_train = data.drop(columns='other')
        print(x_train)


if __name__ == "__main__":
    # Demo.test_chain()
    # print('==='*10)
    # print(Demo.__doc__)
    # print('==='*10)
    # test=Demo('a')
    # print(Demo.__str__())
    # print(str(test))
    demo = Demo()
    path1=Path(r'D:\download_D\1230_小万内容清洗\0109\a.xlsx')
    path2=Path(r'D:\download_D\1230_小万内容清洗\0109\小万数据清洗_res (1).xlsx')
    demo.lst_test()
    demo.pandas_test()