from prepare_data.prepare_raw.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
import datetime
import time
np.random.seed(1234)


class Get_amazon_data_elec(Get_origin_data_base):

    def __init__(self, init_origin_data):

        super(Get_amazon_data_elec, self).__init__( 'elec')

        root = '/home/zxl/project/MTAM-t2/'
        self.raw_data_path = root+"data/raw_data/amazon_electronics/Electronics.json"
        self.raw_data_path_meta = root+"data/raw_data/amazon_electronics/meta_Electronics.json"

        self.data_path = root+"data/orgin_data/elec.csv"

        if  init_origin_data == True:

            self.get_amazon_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)



    def get_amazon_data(self):


        with open(self.raw_data_path, 'r') as fin:
            #确保字段相同
            resultList = []
            error_n =0
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
                    error_n+=1
                    #self.logger.info("Error！！！！！！！！！！！！")
                    #self.logger.info(e)
            print("total error entries"+str(error_n))





            reviews_Electronics_df = pd.DataFrame(resultList)

        with open(self.raw_data_path_meta, 'r') as fin:
            resultList = []
            for line in fin:
                tempDic = eval(line)
                resultDic = {}
                resultDic["cat_id"] = tempDic["category"][-1]
                resultDic["item_id"] = tempDic["asin"]

                resultList.append(resultDic)

            meta_df = pd.DataFrame(resultList)

        reviews_Electronics_df = pd.merge(reviews_Electronics_df, meta_df, on="item_id")
        print(reviews_Electronics_df.shape)

        reviews_Electronics_df = self.filter(reviews_Electronics_df)


        reviews_Electronics_df.to_csv(self.data_path, index=False, encoding="UTF8")
        self.origin_data =  reviews_Electronics_df
        print("well done!!")





