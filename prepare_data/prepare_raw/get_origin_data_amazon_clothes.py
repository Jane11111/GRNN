from prepare_data.prepare_raw.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
import traceback
import datetime
import time
np.random.seed(1234)


class Get_amazon_data_clothes(Get_origin_data_base):

    def __init__(self,init_origin_data,data_name):

        super(Get_amazon_data_clothes, self).__init__(data_name=data_name)
        root = '/home/zxl/project/MTAM-t2/'

        self.raw_data_path = root+"data/raw_data/amazon_"+data_name+"/Clothing_Shoes_and_Jewelry.json"
        self.raw_data_path_meta = root+"data/raw_data/amazon_"+data_name+"/meta_Clothing_Shoes_and_Jewelry.json"
        self.data_path = root+"data/orgin_data/"+data_name+".csv"

        if  init_origin_data == True:

            self.get_amazon_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)



    def get_amazon_data(self):


        with open(self.raw_data_path, 'r') as fin:
            #确保字段相同
            resultList = []
            for line in fin:
                try:
                    line = line.replace("true","True")
                    line = line.replace("false", "False")

                    tempDic = eval(line)
                    resultDic = {}
                    resultDic["user_id"] = tempDic["reviewerID"]
                    resultDic["item_id"] = tempDic["asin"]
                    resultDic["time_stamp"] = tempDic["unixReviewTime"]

                    resultList.append(resultDic)
                except Exception as e:
                    print("Error！！！！！！！！！！！！")
                    print(e)
                    traceback.print_exc()

            reviews_Electronics_df = pd.DataFrame(resultList)


        with open(self.raw_data_path_meta, 'r') as fin:
            resultList = []
            for line in fin:
                try:
                    line = line.replace("false", "False")
                    tempDic = eval(line)
                    resultDic = {}

                    # if "category" in tempDic.keys() and len(tempDic['category']) > 0:
                    #     resultDic["cat_id"] = tempDic["category"][-1]
                    # else:
                    resultDic["cat_id"] = "none"

                    resultDic["item_id"] = tempDic["asin"]
                    resultList.append(resultDic)

                except Exception as e:
                    print("Error！！！！！！！！！！！！")
                    print(e)
                    traceback.print_exc()


            meta_df = pd.DataFrame(resultList)

        reviews_beauty_df = pd.merge(reviews_Electronics_df, meta_df, on="item_id")

        reviews_beauty_df = self.filter(reviews_beauty_df)


        reviews_beauty_df.to_csv(self.data_path, index=False, encoding="UTF8")
        self.origin_data =  reviews_beauty_df



