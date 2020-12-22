"""
Coded by Wenxian and Yinglong
Reviewed by Wendy
"""
import pandas as pd
import os
from datetime import datetime
from sklearn import preprocessing
class Get_origin_data_base():

    '''
    Get_origin_data(type)
    process raw data and get original data. Input dataset name.
    Do statisitcs.
    '''


    def __init__(self,data_name ):
        self.data_name = data_name





    #进行两者统计
    def getDataStatistics(self):

        reviews_df = self.origin_data
        user = set(reviews_df["user_id"].tolist())
        print("The user count is " + str(len(user)))
        item = set(reviews_df["item_id"].tolist())
        print("The item count is " + str(len(item)))
        category = set(reviews_df["cat_id"].tolist())
        print("The category  count is " + str(len(category)))

        behavior = reviews_df.shape[0]
        print("The behavior count is " + str(behavior))


        behavior_per_user = reviews_df.groupby(by=["user_id"], as_index=False)["item_id"].count()
        behavior_per_user = behavior_per_user["item_id"].mean()
        print("The avg behavior of each user count is " + str(behavior_per_user))

        behavior_per_item = reviews_df.groupby(by=["item_id"], as_index=False)["user_id"].count()
        behavior_per_item = behavior_per_item["user_id"].mean()
        print("The avg behavior of each item count is " + str(behavior_per_item))


    def top_user_purchase(self):
        result_list = []

        def get_top_item(x):
            print(x)

        self.origin_data.groupby(by=["user_id","item_id"], as_index=False).apply(lambda x:get_top_item(x))

    def filter(self, data):

        item_filter = data.groupby("item_id").count()

        item_filter = item_filter[item_filter['user_id'] >= 30]

        data = data[
            data['item_id'].isin(item_filter.index)]
        # filtering user < 2
        user_filter = data.groupby("user_id").count()
        if self.data_name == 'elec' or self.data_name == 'order' \
                or self.data_name == 'movie_tv' or self.data_name == 'books':
            #user_filter = user_filter[user_filtermovielen['item_id'] >= 20]
            user_filter = user_filter[user_filter['item_id'] >= 20]
        else :
            user_filter = user_filter[user_filter['item_id'] >= 10]
        data = data[data['user_id'].isin(user_filter.index)]
        return data








