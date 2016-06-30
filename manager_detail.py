# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:04:51 2016

@author: lixiaolong
"""

import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.font_manager as ft_mnger
from scipy import stats

def main():
    """
    read sql data
    and draw some picture
    and output some excel sheets
    ManagerID: string
    """
    ManagerID_list = ['洪流19990101']    
    # ManagerID_list = ['曹名长19970101', '周应波20100101','左金保20110101']    
    # ManagerID_list = ['陈晓翔20010101', '王培20070101', '任泽松20100101', '欧阳沁春20010101', '顾耀强20040101']
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
    # read index data from JYDB
    # mkt
    sql_mktbeta = """
        SELECT A.SecuCode, B.TradingDay, B.PrevClosePrice, B.ClosePrice, A.ChiName, A.InnerCode
        FROM [JYDB].[dbo].[SecuMain] A, [JYDB].[dbo].[QT_IndexQuote] B
        WHERE A.InnerCode = B.InnerCode AND A.SecuCode IN ('000300','000905','000852')
        AND B.ChangePCT is not null
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
        ORDER BY A.SecuCode, B.TradingDay"""
    data_indubeta = pd.read_sql(sql_indubeta, cnxn_jydb, index_col='TradingDay')
    data_indubeta = indexdata_reshape(data_indubeta)
    for ManagerID in ManagerID_list:
        # basic
        sql_manager_basic = """
        SELECT EndDate, SecuAbbr, InvestAdvisorAbbrName
        FROM [jrgcb].[dbo].[FundAndManagerData]
        WHERE ManagerID = '""" + ManagerID + """'
        ORDER BY SecuAbbr, EndDate
        """
        data_manager_basic = pd.read_sql(sql_manager_basic, cnxn_jrgcb)
        fundduration_df = fund_duration(data_manager_basic)
        fundduration_df.to_excel(ManagerID + '_basic_info.xlsx', sheet_name = 'basic_info', index = False)
        # analysis
        sql_manager_analysis = """
        SELECT *
        FROM [jrgcb].[dbo].[FundManagerAnalysis]
        WHERE [ManagerID] = '""" + ManagerID + """'
        ORDER BY [EndDate]
        """
        data_manager_analysis = pd.read_sql(sql_manager_analysis, cnxn_jrgcb, index_col='EndDate')
        data_manager_analysis = data_manager_analysis.dropna(axis = 0, how = 'any')
        data_manager_analysis.to_excel(ManagerID + '_analysis.xlsx', sheet_name = 'analysis')
        # draw mkt indu area
        mkt_beta_df, indu_beta_df, fig_mkt, fig_indu = draw_mkt_indu_area(data_manager_analysis)
        mkt_beta_df.to_excel(ManagerID + '_mkt_beta.xlsx', sheet_name = 'mkt_beta')
        indu_beta_df.to_excel(ManagerID + '_indu_beta.xlsx', sheet_name = 'indu_beta')
        fig_mkt.savefig(ManagerID + '_mkt_area.png', bbox_inches = 'tight')
        fig_indu.savefig(ManagerID + '_indu_area.png', bbox_inches = 'tight')
        # draw alpha bret hist
        fig_mkt, fig_indu = draw_alpha_bret_hist(data_manager_analysis)
        fig_mkt.savefig(ManagerID + '_mkt_hist.png', bbox_inches = 'tight')
        fig_indu.savefig(ManagerID + '_indu_hist.png', bbox_inches = 'tight')
        # draw ts
        fig = draw_ret_ts(data_manager_analysis, data_mktbeta, data_indubeta)
        fig.savefig(ManagerID + '_cumret_plot.png', bbox_inches = 'tight')







def draw_mkt_indu_area(data_manager_analysis):
    """
    draw mkt and indu beta area
    use plt.stackplot()
    info:
        1. beta, stack area
        2. ann alpha, line plot
        3. ann bias ret, line plot
    """
    # split by mkt and indu
    mkt_beta_df, indu_beta_df = split_beta(data_manager_analysis)
    # font
    myfont = ft_mnger.FontProperties(fname='C:/Windows/Fonts/msyh.ttf')
    pylab.mpl.rcParams['axes.unicode_minus'] = False
    # style
    plt.style.use('bmh')
    # x
    x = data_manager_analysis.index.values # end date
    # mkt
    y_area = np.transpose(mkt_beta_df.values)
    y_line = np.transpose([data_manager_analysis.intercept_mkt.values * 250,
              data_manager_analysis.bias_ret_mkt])
    fig_mkt = plt.figure(figsize = (10, 7.5))
    ax_area = fig_mkt.add_subplot(111)
    ax_area.stackplot(x, y_area, alpha = 0.5)
    ax_area.set_ylabel('mkt beta')
    ax_area.legend(mkt_beta_df.columns.values.tolist(),
                   loc = 2,
                   prop = myfont,
                   bbox_to_anchor = (1.11, 1))
    ax_line = ax_area.twinx()
    ax_line.plot(x, y_line)
    ax_line.set_ylabel('alpha & bia ret')
    ax_line.legend(['alpha', 'bias ret'], loc = 2)
    # fig_mkt.savefig('20.png', bbox_inches = 'tight')
    # indu
    y_area = np.transpose(indu_beta_df.values)
    y_line = np.transpose([data_manager_analysis.intercept_indu.values * 250,
              data_manager_analysis.bias_ret_indu])
    fig_indu = plt.figure(figsize = (10, 7.5))
    ax_area = fig_indu.add_subplot(111)
    ax_area.stackplot(x, y_area, alpha = 0.5)
    ax_area.set_ylabel('indu beta')
    ax_area.legend(indu_beta_df.columns.values.tolist(),
                   loc = 2,
                   prop = myfont,
                   bbox_to_anchor = (1.11, 1))
    ax_line = ax_area.twinx()
    ax_line.plot(x, y_line)
    ax_line.set_ylabel('alpha & bia ret')
    ax_line.legend(['alpha', 'bias ret'], loc = 2)
    # return
    return mkt_beta_df, indu_beta_df, fig_mkt, fig_indu

def draw_alpha_bret_hist(data_manager_analysis):
    """
    draw alpha and bret hist
    use plt.hist()
    info:
    1. alpha hist
    2. bret hist
    3. alpha: mean std skew
    4. bret: mean std skew
    """
    # style
    plt.style.use('bmh')
    # parameters
    num_bins = 50
    alpha = 0.3
    histtype = 'stepfilled'
    # mkt
    fig_mkt = plt.figure(figsize = (10, 7.5))
    ax_mkt = fig_mkt.add_subplot(111)
    x_alpha = data_manager_analysis.intercept_mkt.values * 250
    x_bret = data_manager_analysis.bias_ret_mkt.values
    ax_mkt.hist(x_alpha, num_bins,
             normed = True,
             alpha = alpha,
             histtype = histtype,
             facecolor = 'red',
             label = 'alpha mkt')
    ax_mkt.hist(x_bret, num_bins,
             normed = True,
             alpha = alpha,
             histtype = histtype,
             facecolor = 'green',
             label = 'bias ret mkt')
    n_a, (min_a, max_a), m_a, v_a, s_a, k_a = stats.describe(x_alpha)
    n_b, (min_b, max_b), m_b, v_b, s_b, k_b = stats.describe(x_bret)
    ax_mkt.table(cellText = [np.round([n_a, min_a, max_a, m_a, 250**0.5 * v_a, s_a, k_a],3).tolist(),
                                      np.round([n_b, min_b, max_b, m_b, 250**0.5 * v_b, s_b, k_b],3).tolist()],
                 rowLabels = ['alpha mkt', 'bias ret mkt'],
                 colLabels = ['Count', 'Min', 'Max', 'Mean', 'Variance', 'Skew', 'Kurtosis'],
                 loc = 'bottom',
                 bbox = [0, -0.25, 1, 0.15])
    ax_mkt.legend(loc = 2)
    # indu
    fig_indu = plt.figure(figsize = (10, 7.5))
    ax_indu = fig_indu.add_subplot(111)
    x_alpha = data_manager_analysis.intercept_indu.values * 250
    x_bret = data_manager_analysis.bias_ret_indu.values
    ax_indu.hist(x_alpha, num_bins,
             normed = True,
             alpha = alpha,
             histtype = histtype,
             facecolor = 'red',
             label = 'alpha indu')
    ax_indu.hist(x_bret, num_bins,
             normed = True,
             alpha = alpha,
             histtype = histtype,
             facecolor = 'green',
             label = 'bias ret indu')
    n_a, (min_a, max_a), m_a, v_a, s_a, k_a = stats.describe(x_alpha)
    n_b, (min_b, max_b), m_b, v_b, s_b, k_b = stats.describe(x_bret)
    ax_indu.table(cellText = [np.round([n_a, min_a, max_a, m_a, 250**0.5 * v_a, s_a, k_a],3).tolist(),
                                      np.round([n_b, min_b, max_b, m_b, 250**0.5 * v_b, s_b, k_b],3).tolist()],
                 rowLabels = ['alpha indu', 'bias ret indu'],
                 colLabels = ['Count', 'Min', 'Max', 'Mean', 'Variance', 'Skew', 'Kurtosis'],
                 loc = 'bottom',
                 bbox = [0, -0.25, 1, 0.15])
    ax_indu.legend(loc = 2)
    return fig_mkt, fig_indu

def draw_ret_ts(data_manager_analysis, data_mktbeta, data_indubeta):
    """
    draw ret ts comparing with mkt_sync and indu_sync w/o intercept
    use plt.plot()
    info:
    1. manager ret
    2. mkt_sync ret
    3. indu_sync_ret
    4. stats on manager ret
    """
    # style
    plt.style.use('bmh')
    # calc
    ret_ts = data_manager_analysis.Ret.values
    cumret_ts = (ret_ts + 1).cumprod()
    x = data_manager_analysis.index.values
    # use mkt beta and indu beta to follow cumret
    sync_df = pd.DataFrame(index = x, columns = ['mkt_sync', 'indu_sync'])
    for index,row in data_manager_analysis.iterrows():
        # mkt
        if index in data_mktbeta.index:
            mkt_sync = (row.beta_mkt1 * data_mktbeta.ix[index, row.name_mkt1] +
                        row.beta_mkt2 * data_mktbeta.ix[index, row.name_mkt2] +
                        row.beta_mkt3 * data_mktbeta.ix[index, row.name_mkt3])
        else:
            mkt_sync = 0
        sync_df.ix[index, 'mkt_sync'] = mkt_sync
        # indu
        if index in data_indubeta.index:
            indu_sync = (row.beta_indu1 * data_indubeta.ix[index, row.name_indu1] +
                         row.beta_indu2 * data_indubeta.ix[index, row.name_indu2] +
                         row.beta_indu3 * data_indubeta.ix[index, row.name_indu3])
        else:
            indu_sync = 0
        sync_df.ix[index, 'indu_sync'] = indu_sync
    cummkt_ts = (sync_df.mkt_sync.values + 1).cumprod()
    cumindu_ts = (sync_df.indu_sync.values + 1).cumprod()
    y = np.transpose([cumret_ts, cummkt_ts, cumindu_ts])
    # plot
    fig = plt.figure(figsize = (10, 7.5))
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_ylabel('cum return')
    ax.legend(['Manager CumRet', 'Mkt CumRet', 'Indu CumRet'], loc = 2)
    # stats
    n, (mini, maxi), m, v, s, k = stats.describe(ret_ts)
    # max drawdown
    i = np.argmax(np.maximum.accumulate(cumret_ts) - cumret_ts) # end of the period
    j = np.argmax(cumret_ts[:i]) # start of period
    plt.plot([x[i], x[j]], [cumret_ts[i], cumret_ts[j]], 'o', markersize=10)
    maxdd = 1 - cumret_ts[i] / cumret_ts[j]
    # add table
    ax.table(cellText = [np.round([n, mini, maxi, m, 250**0.5 * v, s, k, maxdd],3).tolist()],
             rowLabels = ['Manager Stats'],
             colLabels = ['Count', 'Min', 'Max', 'Mean', 'Variance', 'Skew', 'Kurtosis', 'MaxDD'],
             loc = 'bottom',
             bbox = [0, -0.25, 1, 0.15])
    return fig

def fund_duration(data_manager_basic):
    """
    given data_manager_basic
    EndDate | SecuAbbr | InvestAdvisorAbbrName
    order by enddate
    return table
    | fund name | company | start time | end time |
    """
    fund_unq = data_manager_basic.SecuAbbr.unique()
    cols = ['FundAbbr', 'InvestAdvisor', 'StartTime', 'EndTime']
    result_df = pd.DataFrame(columns = cols)
    for fund in fund_unq:
        temp = data_manager_basic.ix[data_manager_basic.SecuAbbr == fund]
        sd = temp.EndDate.iloc[0]
        ed = temp.EndDate.iloc[-1]
        result_df = result_df.append(dict(zip(cols,[fund,
                                        temp.InvestAdvisorAbbrName.iloc[0],
                                        sd,ed])),
                                        ignore_index = True)
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

def split_beta(data_manager_analysis):
    """
    re-organize beta:
    mkt:
    date | mkt_name1 | mkt_name2 | mkt_name3
    indu:
    date | indu1 | indu2 | indu3 | indu4 | ...
    """
    # time
    date_array = data_manager_analysis.index.values
    # mkt
    mktname_unq = np.append(data_manager_analysis.name_mkt1.unique(),
                            data_manager_analysis.name_mkt2.unique())
    mktname_unq = np.append(mktname_unq,data_manager_analysis.name_mkt3.unique())
    mktname_unq = np.unique(mktname_unq)
    mkt_beta_df = pd.DataFrame(columns = mktname_unq, index = date_array)
    # indu
    induname_unq = np.append(data_manager_analysis.name_indu1.unique(),
                            data_manager_analysis.name_indu2.unique())
    induname_unq = np.append(induname_unq,data_manager_analysis.name_indu3.unique())
    induname_unq = np.unique(induname_unq)
    indu_beta_df = pd.DataFrame(columns = induname_unq, index = date_array)
    for index,row in data_manager_analysis.iterrows():
        mkt_beta_df.ix[index, row.name_mkt1] = row.beta_mkt1
        mkt_beta_df.ix[index, row.name_mkt2] = row.beta_mkt2
        mkt_beta_df.ix[index, row.name_mkt3] = row.beta_mkt3
        indu_beta_df.ix[index, row.name_indu1] = row.beta_indu1
        indu_beta_df.ix[index, row.name_indu2] = row.beta_indu2
        indu_beta_df.ix[index, row.name_indu3] = row.beta_indu3
    mkt_beta_df = mkt_beta_df.fillna(value = 0)
    indu_beta_df = indu_beta_df.fillna(value = 0)
    return mkt_beta_df, indu_beta_df

if __name__ == '__main__':
    main()
