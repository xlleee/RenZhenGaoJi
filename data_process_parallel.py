# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:04:38 2016

@author: lixiaolong
"""

import pyodbc
import pandas as pd
import numpy as np
import datetime
from sklearn import linear_model
from multiprocessing import Pool

########################################################


def main():
    # JYDB db
    cnxn_jydb = pyodbc.connect("""
        DRIVER={SQL Server};
        SERVER=172.16.7.229;
        DATABASE=jydb;
        UID=sa;
        PWD=sa123456""")
    # JRGCB db
    cnxn_jrgcb = pyodbc.connect("""
        DRIVER={SQL Server};
        SERVER=172.16.7.166;
        DATABASE=jrgcb;
        UID=sa;
        PWD=sa123456""")
    ########################################################
    # read index data for JYDB
    sql_mktbeta = """
        SELECT A.SecuCode, B.TradingDay, B.PrevClosePrice, B.ClosePrice,
        A.ChiName, A.InnerCode
        FROM [JYDB].[dbo].[SecuMain] A, [JYDB].[dbo].[QT_IndexQuote] B
        WHERE A.InnerCode = B.InnerCode AND A.SecuCode IN
        ('000300','000905','000852')
        AND B.ChangePCT is not null
        ORDER BY A.SecuCode, B.TradingDay"""
    data_mktbeta = pd.read_sql(sql_mktbeta, cnxn_jydb, index_col='TradingDay')
    data_mktbeta = indexdata_reshape(data_mktbeta)
    sql_indubeta = """
        SELECT A.SecuCode, B.TradingDay, B.PrevClosePrice, B.ClosePrice,
        A.ChiName, A.InnerCode
        FROM [JYDB].[dbo].[SecuMain] A, [JYDB].[dbo].[QT_IndexQuote] B
        WHERE A.InnerCode = B.InnerCode AND A.SecuCode IN
        ('CI005001','CI005002','CI005003','CI005004','CI005005',
        'CI005006','CI005007','CI005008','CI005009','CI005010',
        'CI005011','CI005012','CI005013','CI005014','CI005015',
        'CI005016','CI005017','CI005018','CI005019','CI005020',
        'CI005021','CI005022','CI005023','CI005024','CI005025',
        'CI005026','CI005027','CI005028','CI005029')
        AND B.ChangePCT is not null
        ORDER BY A.SecuCode, B.TradingDay"""
    data_indubeta = pd.read_sql(
        sql_indubeta, cnxn_jydb, index_col='TradingDay')
    data_indubeta = indexdata_reshape(data_indubeta)
    ########################################################
    ########################################################
    # sql to select distinct fund manager
    sql_allmng = """
        SELECT DISTINCT [ManagerID]
        FROM [jrgcb].[dbo].[FundAndManagerData]
        ORDER BY [ManagerID]
        """
    data_allmng = pd.read_sql(sql_allmng, cnxn_jrgcb)

    # call organize data
    ob_win = 180
    args = [(_id, data_mktbeta, data_indubeta, ob_win) for _id in
            data_allmng_.ManagerID]
    with Pool(2) as pool:
        pool.starmap(organize_data, args)


def organize_data(ManagerID, data_mktbeta, data_indubeta, ob_win):
    """
    give ManagerID (1467 for YaWeiGe), organize data to store in sql
    ob_win: ob window length
    return result_df
    """
    # JYDB db
    cnxn_jydb = pyodbc.connect("""
        DRIVER={SQL Server};
        SERVER=172.16.7.229;
        DATABASE=jydb;
        UID=sa;
        PWD=sa123456""")
    # JRGCB db
    cnxn_jrgcb = pyodbc.connect("""
        DRIVER={SQL Server};
        SERVER=172.16.7.166;
        DATABASE=jrgcb;
        UID=sa;
        PWD=sa123456""")
    print(datetime.datetime.now())
    print('Start organize data of: ' + ManagerID)
    # read mng record data
    sql_allrecord = """
        SELECT [InnerCode], [EndDate], [dailyreturn], [FundsofManager],
        [ManagersofFund]
        FROM [jrgcb].[dbo].[FundAndManagerData]
        WHERE [ManagerID] = '""" + ManagerID + """'
        ORDER BY [EndDate]
        """
    data_allrecord = pd.read_sql(sql_allrecord, cnxn_jrgcb)
    # dict of invest advisor
    fundcode_array = data_allrecord.InnerCode.unique()
    codestr = '('
    for code in fundcode_array:
        if len(codestr) == 1:
            codestr += "'" + str(int(code)) + "'"
        else:
            codestr += ",'" + str(int(code)) + "'"
    codestr += ')'
    sql_investadv = """
        SELECT A.InnerCode, B.InvestAdvisorName
        FROM [JYDB].[dbo].[MF_FundArchives] A,
        [JYDB].[dbo].[MF_InvestAdvisorOutline] B
        WHERE A.InvestAdvisorCode = B.InvestAdvisorCode
        AND A.InnerCode IN """ + codestr
    data_investadv = pd.read_sql(sql_investadv, cnxn_jydb)
    # store
    cols = ['ID', 'EndDate', 'InvestAdvisor', 'ManagerID', 'Return']
    result_df = pd.DataFrame(columns=cols)  # store here
    # 先算出复合收益率
    time_array = data_allrecord.EndDate.unique()
    for date in time_array:
        data_subrecord = data_allrecord[data_allrecord.EndDate == date]
        wgted_ret = 0
        wgt = 0
        InAdv = ''
        for index, row in data_subrecord.iterrows():
            wgt += 1 / row.ManagersofFund
            wgted_ret += row.dailyreturn / row.ManagersofFund
            InAdv = data_investadv.InvestAdvisorName[data_investadv.InnerCode
                                                     == int(row.InnerCode)].values[0]
        ret = wgted_ret / wgt
        if np.isnan(ret):
            continue
        else:
            IDstr = ManagerID + pd.to_datetime(date).strftime('%Y%m%d')
            result_df = result_df.append(dict(zip(cols, [IDstr, date, InAdv, ManagerID, ret])),
                                         ignore_index=True)
    # 业绩分解
    # fetch index data and construct np.array
    addcols = ['beta_mkt1', 'beta_mkt2', 'beta_mkt3',
               'name_mkt1', 'name_mkt2', 'name_mkt3',
               'intercept_mkt', 'score_mkt',
               'beta_indu1', 'beta_indu2', 'beta_indu3',
               'name_indu1', 'name_indu2', 'name_indu3',
               'intercept_indu', 'score_indu']
    newcols = cols + addcols
    result_df = result_df.reindex(columns=newcols)
    idx = 0
    total = len(result_df)
    while idx < len(result_df):
        if idx < ob_win - 1:
            # no enough data
            print('line:' + str(idx) + '/' + str(total) + ', skip')
            idx += 1
            continue
        else:
            # get date
            obdates = result_df.iloc[(idx - ob_win + 1):(idx + 1)].EndDate
            timegap = (obdates.iloc[-1] - obdates.iloc[0]).days
            if timegap / ob_win > 9 / 5:
                # dates not continuous
                print('line:' + str(idx) + '/' + str(total) + ', skip')
                idx += 1
                continue
            else:
                # calc
                mng_ret = result_df.iloc[(idx - ob_win + 1):(idx + 1)].Return
                mkt_ret = data_mktbeta.loc[obdates]
                indu_ret = data_indubeta.loc[obdates]
                # LARS
                mng_ret = mng_ret.values
                mkt_ret = mkt_ret.values
                indu_ret = indu_ret.values
                # remove NaN rows
                isnanrow = np.isnan(mkt_ret[:, 1])
                mng_ret = mng_ret[~isnanrow]
                mkt_ret = mkt_ret[~isnanrow]
                indu_ret = indu_ret[~isnanrow]
                # define mkt model
                model = linear_model.LassoCV(positive=True,
                                             cv=10,  # use 10 fold
                                             selection='random',
                                             fit_intercept=True,
                                             normalize=False)
                # mkt
                model.fit(mkt_ret, mng_ret)  # fit(X, y)
                beta_mkt = model.coef_
                name_mkt = data_mktbeta.columns.values
                sortedidx = np.argsort(beta_mkt)
                result_df.ix[idx, 'beta_mkt1'] = beta_mkt[sortedidx[-1]]
                result_df.ix[idx, 'name_mkt1'] = name_mkt[sortedidx[-1]]
                result_df.ix[idx, 'beta_mkt2'] = beta_mkt[sortedidx[-2]]
                result_df.ix[idx, 'name_mkt2'] = name_mkt[sortedidx[-2]]
                result_df.ix[idx, 'beta_mkt3'] = beta_mkt[sortedidx[-3]]
                result_df.ix[idx, 'name_mkt3'] = name_mkt[sortedidx[-3]]
                result_df.ix[idx, 'intercept_mkt'] = model.intercept_
                result_df.ix[idx, 'score_mkt'] = model.score(mkt_ret, mng_ret)
                # define indu model
                model = linear_model.LassoCV(positive=True,
                                             cv=20,  # use 20 fold
                                             selection='random',
                                             fit_intercept=True,
                                             normalize=False)
                # indu
                model.fit(indu_ret, mng_ret)
                beta_indu = model.coef_
                name_indu = data_indubeta.columns.values
                sortedidx = np.argsort(beta_indu)
                result_df.ix[idx, 'beta_indu1'] = beta_indu[sortedidx[-1]]
                result_df.ix[idx, 'name_indu1'] = name_indu[sortedidx[-1]]
                result_df.ix[idx, 'beta_indu2'] = beta_indu[sortedidx[-2]]
                result_df.ix[idx, 'name_indu2'] = name_indu[sortedidx[-2]]
                result_df.ix[idx, 'beta_indu3'] = beta_indu[sortedidx[-3]]
                result_df.ix[idx, 'name_indu3'] = name_indu[sortedidx[-3]]
                result_df.ix[idx, 'intercept_indu'] = model.intercept_
                result_df.ix[idx, 'score_indu'] = model.score(
                    indu_ret, mng_ret)
                idx += 1
                print('line:' + str(idx) + '/' + str(total) + ', done')
    # end of while loop
    print(datetime.datetime.now())
    print('End organize data of: ' + ManagerID)
    result_df.to_excel(ManagerID + 'ob' + str(ob_win) + '.xlsx')


def indexdata_reshape(indexdata_df):
    """
    reshape indexdata:
    col by ChiName
    row by TradingDay
    data: daily return
    """
    chiname_array = indexdata_df.ChiName.unique()
    reshaped_df = pd.DataFrame()
    for name in chiname_array:
        temp = indexdata_df[indexdata_df.ChiName == name].ClosePrice \
            / indexdata_df[indexdata_df.ChiName == name].PrevClosePrice - 1
        data = pd.DataFrame(data=temp.values,
                            columns=[name],
                            index=temp.index)
        reshaped_df = reshaped_df.join(data, how='outer')
    return reshaped_df.dropna(axis=0, how='any')


if __name__ == '__main__':
    main()
