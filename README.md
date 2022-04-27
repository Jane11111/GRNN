# OAG-LC

This is a pytorch implementation of the OAG-LC (Order-Aware Graph Neural Network for Sequential Recommendation) and baseline methods.



# Requirements

- python 3.7.9
- pytorch 1.4.0
- pandas 1.1.2

# How to Run

1. preprocess the datasets:

​      put the datasets to ` RNN/data/origin_data/`

​      ` python main_preparedata.py`

2. train our proposed model using `main_best.py`

   `python main_best.py`
   
   you can change the `model` variable in `main_best.py` to choose which model you want to run. `GRNN` is our proposed `OAG-LC` model.

3. train the baseline models using `main_baseline_best.py` and `main_stargnn.py`

   `python main_baseline_best.py`

   You can change the `model` variable in `main_baseline_best.py` to choose which baseline model you want to run. The optional variables are `[GRU4Rec, NARM, SASRec, STAMP, SRGNN, GCSAN,LESSR]`

   `python main_stargnn.py`

   the `main_stargnn.py` is used to train the SGNN-HN model

# Parameter Settings

The pre-trained OAG-LC models  for each dataset is located at `GRNN/data/pretrained_models`.  

The corresponding parameters are listed as below.

| 数据集       | learning_rate | dropout_prob | agg_layer |
| ------------ | ------------- | ------------ | --------- |
| Electronics  | 0.005         | 0.25         | 1         |
| Tmall        | 0.001         | 0            | 1         |
| Movies&TV    | 0.005         | 0            | 1         |
| Home&kitchen | 0.005         | 0.25         | 3         |



The other parameters for our model are set as the default value for all the dataset, which is described in `./config/default.yaml`
