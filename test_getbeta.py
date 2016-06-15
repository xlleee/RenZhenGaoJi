# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:38:15 2016

@author: lixiaolong
"""

from WindPy import *
import numpy as np
from sklearn import linear_model

w.start()

starttime = "2015-01-01"
endtime = "2016-06-13"

indexcode = "950096.CSI"

hs300code = "000300.SH"
zz500code = "000905.SH"
sz50code = "000016.SH"

index_data = w.wsd(indexcode, "pct_chg", starttime, endtime, "")
hs300_data = w.wsd(hs300code, "pct_chg", starttime, endtime, "")
zz500_data = w.wsd(zz500code, "pct_chg", starttime, endtime, "")
sz50_data = w.wsd(sz50code, "pct_chg", starttime, endtime, "")

index_data = index_data.Data[0]
hs300_data = hs300_data.Data[0]
zz500_data = zz500_data.Data[0]
sz50_data = sz50_data.Data[0]
# fit
model = linear_model.LassoCV(positive = True,
                            cv = int(len(index_data)/60), 
                            selection = 'random',
                            fit_intercept = True,
                            normalize = False)
model.fit(np.transpose([hs300_data, zz500_data, sz50_data]), index_data) # fit(X, y)
sstr = 'index ret = %6.4f * hs300 + %6.4f * zz500 + %6.4f * sz50 + %6.4f, rsqr = %6.4f'
result = sstr % (model.coef_[0], model.coef_[1], model.coef_[2], model.intercept_, 
                 model.score(np.transpose([hs300_data, zz500_data, sz50_data]), index_data))
