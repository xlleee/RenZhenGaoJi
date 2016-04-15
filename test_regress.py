# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:44:21 2016

@author: lixiaolong
"""

from WindPy import *
import numpy as np
from sklearn import linear_model

w.start()
# index
index_str = """H30164.CSI,H30165.CSI,H30169.CSI,H30170.CSI,H30171.CSI,H30172.CSI,H30173.CSI,H30174.CSI,H30175.CSI,H30176.CSI,H30177.CSI,H30178.CSI,H30179.CSI,H30181.CSI,H30182.CSI,H30183.CSI,H30184.CSI,H30185.CSI,H30186.CSI"""
index_data = w.wsd(index_str,
                   "close",
                   "2016-02-10",
                   "2016-04-10",
                   "PriceAdj=F")
index_data_ts = index_data.Data
index_data_ts_array = np.array(index_data_ts).transpose()
index_ret = index_data_ts_array[1:,:] / index_data_ts_array[0:-1,:] - 1
# fund
zesong_data = w.wsd("590008.OF", 
                    "nav_adj", 
                    "2016-02-10", 
                    "2016-04-10", 
                    "PriceAdj=F")
zesong_data_ts = zesong_data.Data
zesong_data_ts_array = np.array(zesong_data_ts).transpose()
zesong_ret = zesong_data_ts_array[1:,:] / zesong_data_ts_array[0:-1,:] - 1
# LARS
clf = linear_model.Lars(n_nonzero_coefs=5,positive=True)
clf.fit(index_ret,zesong_ret)
coef = clf.coef_
intercept = clf.intercept_[0]