# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 17:14:46 2016

@author: lixiaolong
"""
from WindPy import *
import statsmodels.api as sm
import pandas as pd
import numpy as np

w.start()
starttime = "2014-10-01 09:30:00"
endtime = "2014-12-31 15:00:00"
constitime = "20140101"
indexretcode = "000905.SH"
indexconcode = "000926.SH"
barSize = "BarSize=60"
# read data
# constituent
arg = "date=" + constitime + ";windcode=" + indexconcode
indexconsti = w.wset("sectorconstituent",arg)
stockcode = indexconsti.Data[1]
# index ret
indexret = w.wsi(indexretcode, "pct_chg", starttime, endtime, barSize)
indexret_ts = indexret.Data[0]
# granger parameters
window = 20
setkind = 'f'
store = pd.DataFrame(columns=['code','conclusion','crit_value',
'df','pvalue','signif','statistic'])
for code in stockcode:
    stockret = w.wsi(code, "pct_chg", starttime, endtime, barSize)
    stockret_ts = stockret.Data[0]
    if stockret_ts.count(0)/len(stockret_ts) > 0.9 or sum(np.isnan(stockret_ts)) > 0:
        continue
    else:
        vardata = pd.DataFrame()
        vardata[indexretcode] = indexret_ts
        vardata[code] = stockret_ts
        model = sm.tsa.VAR(vardata.values)
        result = model.fit(window)
        rdd = result.test_causality(1,0,kind=setkind)
        temp = pd.DataFrame({'code': code,
        'conclusion': rdd['conclusion'],
        'crit_value': rdd['crit_value'],
        'df': [rdd['df']],
        'pvalue': rdd['pvalue'],
        'signif': rdd['signif'],
        'statistic':rdd['statistic']})
        store = store.append(temp)
# output
filename = indexconcode[0:6] + '_vs_' + indexretcode[0:6] + '_' + starttime[0:10] + '_to_' \
+ endtime[0:10] + '.xlsx'
store.to_excel(filename)
