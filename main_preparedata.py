# -*- coding: utf-8 -*-
# @Time    : 2020-12-16 10:55
# @Author  : zxl
# @FileName: main_preparedata.py

from prepare_data.prepare_raw.get_origin_data_fs import Get_fs_data
from prepare_data.prepare_raw.get_origin_data_amazon_kindle import Get_amazon_data_kindle
from prepare_data.prepare_raw.get_origin_data_amazon_clothes import Get_amazon_data_clothes
from prepare_data.prepare_raw.get_origin_data_amazon_sports import Get_amazon_data_sports
from prepare_data.prepare_raw.get_origin_data_amazon_phone import Get_amazon_data_phone
from prepare_data.prepare_raw.get_origin_data_amazon_books import Get_amazon_data_books
from prepare_data.prepare_raw.get_origin_data_amazon_home import Get_amazon_data_home
from prepare_data.prepare_raw.get_origin_data_amazon_music import Get_amazon_data_music
from prepare_data.prepare_raw.get_origin_data_amazon_elec import Get_amazon_data_elec
from prepare_data.prepare_raw.get_origin_data_amazon_movie_tv import Get_amazon_data_movie_tv
from prepare_data.prepare_raw.get_origin_data_tmall_buy import Get_tmall_buy_data
from prepare_data.split_train_test import prepare_data_base
if __name__ == '__main__':

    data_name = 'movie_tv'
    init_origin_data = False
    length_of_user_history = 100

    print(data_name)

    if data_name == 'fs':
        origin_data_model = Get_fs_data(init_origin_data,data_name)
    elif data_name == 'kindle':
        origin_data_model = Get_amazon_data_kindle(init_origin_data,data_name)
    elif data_name == 'clothes':
        origin_data_model = Get_amazon_data_clothes(init_origin_data,data_name)
    elif data_name == 'sports':
        origin_data_model = Get_amazon_data_sports(init_origin_data,data_name)
    elif data_name == 'phone':
        origin_data_model = Get_amazon_data_phone(init_origin_data,data_name)
    elif data_name == 'books':
        origin_data_model = Get_amazon_data_books(init_origin_data,data_name)
    elif data_name == 'home':
        origin_data_model = Get_amazon_data_home(init_origin_data,data_name)
    elif data_name == 'music':
        origin_data_model = Get_amazon_data_music(init_origin_data,data_name)
    elif data_name == 'elec':
        origin_data_model = Get_amazon_data_elec(init_origin_data,data_name)
    elif data_name == 'movie_tv':
        origin_data_model = Get_amazon_data_movie_tv(init_origin_data)
    elif data_name == 'tmall_buy':
        origin_data_model = Get_tmall_buy_data(init_origin_data)


    origin_data_df = origin_data_model.origin_data
    origin_data_model.getDataStatistics()

    train_test_model = prepare_data_base(data_name=data_name,
                                         origin_data=origin_data_df,
                                         length_of_user_history=length_of_user_history)

    train_set,test_set,dev_set = train_test_model.get_train_test()


