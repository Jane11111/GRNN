from prepare_data.prepare_raw.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
np.random.seed(1234)


class Get_movie_data(Get_origin_data_base):

    def __init__(self,init_origin_data):

        super(Get_movie_data, self).__init__('movielen')
        root = '/home/zxl/project/MTAM-t2/'

        self.data_path = "data/orgin_data/movielen.csv"

        if  init_origin_data == True:
            self.movie_data = pd.read_csv("data/raw_data/ml-20m/movies.csv")
            self.ratings_data = pd.read_csv("data/raw_data/ml-20m/ratings.csv")
            self.get_movie_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)



    def get_movie_data(self):


        print(self.ratings_data.shape)
        user_filter = self.ratings_data.groupby("userId").count()
        userfiltered = user_filter.sample(frac=0.05)
        self.ratings_data = self.ratings_data[self.ratings_data['userId'].isin(userfiltered.index)]
        print(self.ratings_data.shape)

        #进行拼接，进行格式的规范化
        self.origin_data = pd.merge(self.ratings_data,self.movie_data,on="movieId")
        self.origin_data = self.origin_data[["userId","movieId","timestamp","genres"]]
        self.origin_data = self.origin_data.rename(columns={"userId": "user_id",
                                         "movieId": "item_id",
                                         "timestamp":"time_stamp",
                                         "genres":"cat_id",
                                         })


        self.filtered_data = self.filter(self.origin_data)
        self.filtered_data.to_csv(self.data_path, encoding="UTF8", index=False)
        self.origin_data = self.filtered_data













