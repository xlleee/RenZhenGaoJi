# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 13:04:38 2016
based on new db
@author: lixiaolong
"""

import pyodbc
import pandas as pd
import numpy as np
import datetime
from sklearn import linear_model

########################################################
def main():
    """
    startdatestr: 开始日期 yyyy-mm-dd
    enddatestr: 结束日期 yyyy-mm-dd
    ob_win: 观察窗口
    要根据ob_win来算一个取数据的时间窗口
    """
    startdatestr = 'yyyy-mm-dd'
    enddatestr = 'yyyy-mm-dd'
    ob_win = 90
    sdtime = datetime.datetime.strptime(startdatestr, '%Y-%m-%d')
    # 按照自然日/工作日 = 7/5，然后再多加30天，应该能够覆盖ob_win的长度
    td = datetime.timedelta(days = ob_win * 7 / 5 + 30)
    sdtime = sdtime - td
    rsdt_str = sdtime.strftime('%Y-%m-%d')
    redt_str = enddatestr
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
    cursor_jrgcb = cnxn_jrgcb.cursor() # cursor to execute sql strings
    # create sqlalchemy for writing
    ########################################################
    # read index data from JYDB
    # mkt
    sql_mktbeta = """
        SELECT A.SecuCode, B.TradingDay, B.PrevClosePrice, B.ClosePrice, A.ChiName, A.InnerCode
        FROM [JYDB].[dbo].[SecuMain] A, [JYDB].[dbo].[QT_IndexQuote] B
        WHERE A.InnerCode = B.InnerCode AND A.SecuCode IN ('000300','000905','000852')
        AND B.ChangePCT is not null
        AND B.TradingDay >= '""" + rsdt_str + """' AND B.TradingDay <= '""" + redt_str + """'
        ORDER BY A.SecuCode, B.TradingDay"""
    data_mktbeta = pd.read_sql(sql_mktbeta, cnxn_jydb, index_col='TradingDay')
    data_mktbeta = indexdata_reshape(data_mktbeta)
    # indu
    sql_indubeta = """
        SELECT A.SecuCode, B.TradingDay, B.PrevClosePrice, B.ClosePrice, A.ChiName, A.InnerCode
        FROM [JYDB].[dbo].[SecuMain] A, [JYDB].[dbo].[QT_IndexQuote] B
        WHERE A.InnerCode = B.InnerCode AND A.SecuCode IN
        ('CI005001','CI005002','CI005003','CI005004','CI005005',
        'CI005006','CI005007','CI005008','CI005009','CI005010',
        'CI005011','CI005012','CI005013','CI005014','CI005015',
        'CI005016','CI005017','CI005018','CI005019','CI005020',
        'CI005021','CI005022','CI005023','CI005024','CI005025',
        'CI005026','CI005027','CI005028','CI005029')
        AND B.ChangePCT is not null
        AND B.TradingDay >= '""" + rsdt_str + """' AND B.TradingDay <= '""" + redt_str + """'
        ORDER BY A.SecuCode, B.TradingDay"""
    data_indubeta = pd.read_sql(sql_indubeta, cnxn_jydb, index_col='TradingDay')
    data_indubeta = indexdata_reshape(data_indubeta)
    # read Fund and Manager Data from JRGCB
    sql_FAndMdata = """
        SELECT [ManagerID], [InnerCode], [EndDate], [dailyreturn], [FundsofManager], [ManagersofFund], [InvestAdvisorAbbrName]
        FROM [jrgcb].[dbo].[FundAndManagerData]
        WHERE Enddate >= '""" + rsdt_str + """' AND Enddate <= '""" + redt_str + """'
        ORDER BY [ManagerID], [EndDate]
        """
    data_FAndMdata = pd.read_sql(sql_FAndMdata, cnxn_jrgcb)
    allmnglist = data_FAndMdata.ManagerID.unique().tolist() # get all mng list
    # create table
    create_table(cursor_jrgcb)
    ########################################################
    # call organize data
    for ManagerID in allmnglist:
        result_df = organize_data(ManagerID,
                                  data_mktbeta,
                                  data_indubeta,
                                  data_FAndMdata,
                                  startdatestr, enddatestr, ob_win)
        # result_df.to_excel(ManagerID + 'ob' + str(ob_win) + '.xlsx')
        write_all_sql = """
        INSERT INTO [jrgcb].[dbo].[FundManagerAnalysis]
        VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        write_withnull_sql = """
        INSERT INTO [jrgcb].[dbo].[FundManagerAnalysis]
        ([ID],[EndDate],[InvestAdvisor],[ManagerID],[Ret])
        VALUES
        (?,?,?,?,?)
        """
        for row in result_df.values.tolist():
            if np.isnan(row[5]):
                cursor_jrgcb.execute(write_withnull_sql,row[0:5])
                cursor_jrgcb.commit()
            else:
                cursor_jrgcb.execute(write_all_sql,row)
                cursor_jrgcb.commit()






def organize_data(ManagerID, data_mktbeta, data_indubeta, data_FAndMdata, startdatestr, enddatestr, ob_win):
    """
    give ManagerID, organize data to store in sql
    ob_win: ob window length
    return result_df
    """
    # print(datetime.datetime.now())
    # print('Start organize data of: ' + ManagerID)
    # read mng record data
    data_allrecord = data_FAndMdata[data_FAndMdata.ManagerID == ManagerID]
    # store
    cols = ['ID','EndDate','InvestAdvisor','ManagerID','Ret']
    result_df = pd.DataFrame(columns = cols) # store here
    # 先算出复合收益率
    time_array = data_allrecord.EndDate.unique()
    for date in time_array:
        data_subrecord = data_allrecord[data_allrecord.EndDate == date]
        wgted_ret = 0
        wgt = 0
        InAdv = ''
        for index,row in data_subrecord.iterrows():
            wgt += 1 / row.ManagersofFund
            wgted_ret += row.dailyreturn / row.ManagersofFund
            InAdv = row.InvestAdvisorAbbrName
        ret = wgted_ret / wgt
        if np.isnan(ret):
            continue
        else:
            IDstr = ManagerID + pd.to_datetime(date).strftime('%Y%m%d')
            result_df = result_df.append(dict(zip(cols,[IDstr,date,InAdv,ManagerID,ret])),
                                         ignore_index=True)
    # 业绩分解
    # fetch index data and construct np.array
    addcols = ['beta_mkt1', 'beta_mkt2', 'beta_mkt3',
               'name_mkt1', 'name_mkt2', 'name_mkt3',
               'intercept_mkt', 'score_mkt',
               'bias_ret_mkt', 'bias_var_mkt', 'bias_score_mkt', # bias mkt: 风格的偏离
               'beta_indu1', 'beta_indu2', 'beta_indu3',
               'beta_indu4', 'beta_indu5', 'beta_indu6',
               'beta_indu7', 'beta_indu8', 'beta_indu9',
               'beta_indu10', 'beta_indu11', 'beta_indu12',
               'beta_indu13', 'beta_indu14', 'beta_indu15',
               'beta_indu16', 'beta_indu17', 'beta_indu18',
               'beta_indu19', 'beta_indu20', 'beta_indu21',
               'beta_indu22', 'beta_indu23', 'beta_indu24',
               'beta_indu25', 'beta_indu26', 'beta_indu27',
               'beta_indu28', 'beta_indu29',
               'name_indu1', 'name_indu2', 'name_indu3',
               'name_indu4', 'name_indu5', 'name_indu6',
               'name_indu7', 'name_indu8', 'name_indu9',
               'name_indu10', 'name_indu11', 'name_indu12',
               'name_indu13', 'name_indu14', 'name_indu15',
               'name_indu16', 'name_indu17', 'name_indu18',
               'name_indu19', 'name_indu20', 'name_indu21',
               'name_indu22', 'name_indu23', 'name_indu24',
               'name_indu25', 'name_indu26', 'name_indu27',
               'name_indu28', 'name_indu29',
               'intercept_indu', 'score_indu',
               'bias_ret_indu', 'bias_var_indu', 'bias_score_indu'] # bias indu：行业的偏离
    newcols = cols + addcols
    result_df = result_df.reindex(columns = newcols)
    idx = 0
    # total = len(result_df)
    while idx < len(result_df):
        if idx < ob_win - 1:
            # no enough data
            # print('line:' + str(idx) + '/' + str(total) + ', skip')
            idx += 1
            continue
        else:
            # get date
            obdates = result_df.iloc[(idx-ob_win+1):(idx+1)].EndDate
            timegap = (obdates.iloc[-1] - obdates.iloc[0]).days
            if timegap / ob_win > 9 / 5:
                # dates not continuous
                # print('line:' + str(idx) + '/' + str(total) + ', skip')
                idx += 1
                continue
            else:
                # calc
                mng_ret = result_df.iloc[(idx-ob_win+1):(idx+1)].Ret
                mkt_ret = data_mktbeta.loc[obdates]
                indu_ret = data_indubeta.loc[obdates]
                mng_ret = mng_ret.values
                mkt_ret = mkt_ret.values
                indu_ret = indu_ret.values
                # remove NaN rows
                isnanrow = np.isnan(mkt_ret[:,1])
                mng_ret = mng_ret[~isnanrow]
                mkt_ret = mkt_ret[~isnanrow]
                indu_ret = indu_ret[~isnanrow]
                # define mkt model
                model = linear_model.LassoCV(positive = True,
                                           cv = int(ob_win/30), # subsample size = 30
                                           selection = 'random',
                                           fit_intercept = True,
                                           normalize = False)
                # mkt
                model.fit(mkt_ret, mng_ret) # fit(X, y)
                beta_mkt = model.coef_
                name_mkt = data_mktbeta.columns.values
                sortedidx = np.argsort(beta_mkt)
                result_df.ix[idx,'beta_mkt1'] = beta_mkt[sortedidx[-1]]
                result_df.ix[idx,'name_mkt1'] = name_mkt[sortedidx[-1]]
                result_df.ix[idx,'beta_mkt2'] = beta_mkt[sortedidx[-2]]
                result_df.ix[idx,'name_mkt2'] = name_mkt[sortedidx[-2]]
                result_df.ix[idx,'beta_mkt3'] = beta_mkt[sortedidx[-3]]
                result_df.ix[idx,'name_mkt3'] = name_mkt[sortedidx[-3]]
                result_df.ix[idx,'intercept_mkt'] = model.intercept_
                result_df.ix[idx,'score_mkt'] = model.score(mkt_ret, mng_ret)
                # bias mkt
                # calc ret
                b_avg = np.mean(beta_mkt)
                b_adj = beta_mkt - b_avg
                ct = 1 / np.sum(b_adj[b_adj > 0]) # scale factor
                b_adj = b_adj * ct
                bias_retts_mkt = np.dot(mkt_ret, b_adj) # dot成个加权的收益率
                temp = np.mean(bias_retts_mkt) * 250 # daily ret 的年化
                if np.isnan(temp):
                    temp = 0
                result_df.ix[idx,'bias_ret_mkt'] = temp
                temp = np.std(bias_retts_mkt) * 250 ** 0.5 # daily ret std 的年化
                if np.isnan(temp):
                    temp = 0
                result_df.ix[idx,'bias_var_mkt'] = temp
                # calc score
                # std coef
                result_df.ix[idx,'bias_score_mkt'] = np.std(beta_mkt)
                # define indu model
                model = linear_model.LassoCV(positive = True,
                                           cv = int(ob_win/30), # subsample size = 30
                                           selection = 'random',
                                           fit_intercept = True,
                                           normalize = False)
                # indu
                model.fit(indu_ret, mng_ret)
                beta_indu = model.coef_
                name_indu = data_indubeta.columns.values
                sortedidx = np.argsort(beta_indu)
                for i in range(1,30):
                    # from 1 to 29
                    eval('result_df.ix[idx,"beta_indu' + str(i) + '"] = beta_indu[sortedidx[-' + str(i) + ']]')
                    # which means:
                    # result_df.ix[idx,'beta_indui'] = beta_indu[sortedidx[-i]]
                    eval('result_df.ix[idx,"name_indu' + str(i) + '"] = name_indu[sortedidx[-' + str(i) + ']]')
                    # which means:
                    # result_df.ix[idx,'name_indui'] = name_indu[sortedidx[-i]]
                result_df.ix[idx,'intercept_indu'] = model.intercept_
                result_df.ix[idx,'score_indu'] = model.score(indu_ret, mng_ret)
                # bias indu
                # calc ret
                b_avg = np.mean(beta_indu)
                b_adj = beta_indu - b_avg
                ct = 1 / np.sum(b_adj[b_adj > 0]) # scale factor
                b_adj = b_adj * ct
                bias_retts_indu = np.dot(indu_ret, b_adj) # dot成个加权的收益率
                temp = np.mean(bias_retts_indu) * 250 # daily ret 的年化
                if np.isnan(temp):
                    temp = 0
                result_df.ix[idx,'bias_ret_indu'] = temp
                temp = np.std(bias_retts_indu) * 250 ** 0.5 # daily ret std 的年化
                if np.isnan(temp):
                    temp = 0
                result_df.ix[idx,'bias_var_indu'] = temp
                # calc score
                # std coef
                result_df.ix[idx,'bias_score_indu'] = np.std(beta_indu)
                # end of calc
                idx += 1
                # print('line:' + str(idx) + '/' + str(total) + ', done')
    # end of while loop
    # 截取starttime和endtime之间的result
    sdtime = datetime.datetime.strptime(startdatestr, '%Y-%m-%d')
    edtime = datetime.datetime.strptime(enddatestr, '%Y-%m-%d')
    result_df = result_df[(result_df.EndDate >= sdtime) & (result_df.EndDate <= edtime)]
    # print(datetime.datetime.now())
    print('End organize data of: ' + ManagerID)
    return result_df


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
        data = pd.DataFrame(data = temp.values,
                            columns = [name],
                            index = temp.index)
        reshaped_df = reshaped_df.join(data, how = 'outer')
    return reshaped_df.dropna(axis=0, how='any')

def create_table(cursor_jrgcb):
    """
    create new table
    29 industries all included
    """
    cursor_jrgcb.execute(
        """
       CREATE TABLE [jrgcb].[dbo].[FundManagerAnalysis_v2]
       (ID VARCHAR(255) NOT NULL, EndDate DATETIME NULL, InvestAdvisor VARCHAR(255) NULL,
       ManagerID VARCHAR(255) NULL, Ret FLOAT(53) NULL, beta_mkt1 FLOAT(53) NULL,
       beta_mkt2 FLOAT(53) NULL, beta_mkt3 FLOAT(53) NULL,
       name_mkt1 VARCHAR(255) NULL, name_mkt2 VARCHAR(255) NULL,
       name_mkt3 VARCHAR(255) NULL, intercept_mkt FLOAT(53) NULL,
       score_mkt FLOAT(53) NULL, bias_ret_mkt FLOAT(53) NULL,
       bias_var_mkt FLOAT(53) NULL, bias_score_mkt FLOAT(53) NULL,
       beta_indu1 FLOAT(53) NULL, beta_indu2 FLOAT(53) NULL,
       beta_indu3 FLOAT(53) NULL, beta_indu4 FLOAT(53) NULL,
       beta_indu5 FLOAT(53) NULL, beta_indu6 FLOAT(53) NULL,
       beta_indu7 FLOAT(53) NULL, beta_indu8 FLOAT(53) NULL,
       beta_indu9 FLOAT(53) NULL, beta_indu10 FLOAT(53) NULL,
       beta_indu11 FLOAT(53) NULL, beta_indu12 FLOAT(53) NULL,
       beta_indu13 FLOAT(53) NULL, beta_indu14 FLOAT(53) NULL,
       beta_indu15 FLOAT(53) NULL, beta_indu16 FLOAT(53) NULL,
       beta_indu17 FLOAT(53) NULL, beta_indu18 FLOAT(53) NULL,
       beta_indu19 FLOAT(53) NULL, beta_indu20 FLOAT(53) NULL,
       beta_indu21 FLOAT(53) NULL, beta_indu22 FLOAT(53) NULL,
       beta_indu23 FLOAT(53) NULL, beta_indu24 FLOAT(53) NULL,
       beta_indu25 FLOAT(53) NULL, beta_indu26 FLOAT(53) NULL,
       beta_indu27 FLOAT(53) NULL, beta_indu28 FLOAT(53) NULL,
       beta_indu29 FLOAT(53) NULL,
       name_indu1 VARCHAR(255) NULL, name_indu2 VARCHAR(255) NULL,
       name_indu3 VARCHAR(255) NULL, name_indu4 VARCHAR(255) NULL,
       name_indu5 VARCHAR(255) NULL, name_indu6 VARCHAR(255) NULL,
       name_indu7 VARCHAR(255) NULL, name_indu8 VARCHAR(255) NULL,
       name_indu9 VARCHAR(255) NULL, name_indu10 VARCHAR(255) NULL,
       name_indu11 VARCHAR(255) NULL, name_indu12 VARCHAR(255) NULL,
       name_indu13 VARCHAR(255) NULL, name_indu14 VARCHAR(255) NULL,
       name_indu15 VARCHAR(255) NULL, name_indu16 VARCHAR(255) NULL,
       name_indu17 VARCHAR(255) NULL, name_indu18 VARCHAR(255) NULL,
       name_indu19 VARCHAR(255) NULL, name_indu20 VARCHAR(255) NULL,
       name_indu21 VARCHAR(255) NULL, name_indu22 VARCHAR(255) NULL,
       name_indu23 VARCHAR(255) NULL, name_indu24 VARCHAR(255) NULL,
       name_indu25 VARCHAR(255) NULL, name_indu26 VARCHAR(255) NULL,
       name_indu27 VARCHAR(255) NULL, name_indu28 VARCHAR(255) NULL,
       name_indu29 VARCHAR(255) NULL,
       intercept_indu FLOAT(53) NULL, score_indu FLOAT(53) NULL,
       bias_ret_indu FLOAT(53) NULL, bias_var_indu FLOAT(53) NULL,
       bias_score_indu FLOAT(53) NULL,
       PRIMARY KEY (ID))
       """)
       cursor_jrgcb.commit()

if __name__ == '__main__':
    main()
