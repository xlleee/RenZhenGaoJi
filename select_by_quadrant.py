# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:27:28 2016

@author: lixiaolong
"""

import pyodbc
import pandas as pd
import numpy as np



def main():
    date_str = '2016-05-30'
    cnxn_jrgcb = pyodbc.connect("""
        DRIVER={SQL Server};
        SERVER=172.16.7.166;
        DATABASE=jrgcb;
        UID=sa;
        PWD=sa123456""")
    # select data
    select_data_sql = """
        SELECT *
        FROM [jrgcb].[dbo].[FundManagerAnalysis]
        WHERE [EndDate] =
            (
                SELECT MAX([EndDate])
                FROM [jrgcb].[dbo].[FundManagerAnalysis]
                WHERE [EndDate] <= '""" + date_str + """'
            )
        """
    data_allrecord = pd.read_sql(select_data_sql, cnxn_jrgcb, index_col = 'ManagerID')

    Q1_result = select_Q1(data_allrecord)
    Q1_result.to_excel('Quadrant1_'+date_str+'.xlsx')

    Q2_result = select_Q2(data_allrecord)
    Q2_result.to_excel('Quadrant2_'+date_str+'.xlsx')

    Q3_result = select_Q3(data_allrecord)
    Q3_result.to_excel('Quadrant3_'+date_str+'.xlsx')

    Q4_result = select_Q4(data_allrecord)
    Q4_result.to_excel('Quadrant4_'+date_str+'.xlsx')


def select_Q1(data_allrecord, score_lvl = 0.80, beta_mkt_lvl = 0.60, beta_indu_lvl = 0.30, select_mkt = 0, select_indu = 0):
    """
    data_allrecord: all record
    风格：偏
    行业：偏
    工具型基金
    风格：
    score_mkt >= score_lvl
    某个beta_mkt >= beta_mkt_lvl
    行业：
    score_indu >= score_lvl
    某个beta_indu >= beta_indu_lvl
    后面类似
    筛选标准：
    作为工具型基金，跟踪热点更为重要，看中bias，从bias ret，var，score衡量？
    其次看投资效率，如果不如直接投指数，那要jjjl干啥？
    打分：
    1. bias ret / var
    2. bias score
    3. alpha(intercept)
    """
    if select_mkt != 0:
        # 指定了风格
        data_allrecord = data_allrecord.ix[data_allrecord.name_mkt1 == select_mkt]
    if select_indu != 0:
        # 指定了行业
        data_allrecord = data_allrecord.ix[data_allrecord.name_indu1 == select_indu]
    # beta_mkt and beta_indu >= level
    # score >= level
    data_allrecord = data_allrecord.ix[(data_allrecord.beta_mkt1 >= beta_mkt_lvl) &
                                       (data_allrecord.beta_indu1 >= beta_indu_lvl) &
                                       (data_allrecord.score_mkt >= score_lvl) &
                                       (data_allrecord.score_indu >= score_lvl)]
    # select
    bias_SR_mkt = data_allrecord.bias_ret_mkt.values / data_allrecord.bias_var_mkt.values
    bias_score_mkt = data_allrecord.bias_score_mkt.values
    intercept_mkt = data_allrecord.intercept_mkt.values
    sc_mkt = u_sc_cal([bias_SR_mkt, bias_score_mkt, intercept_mkt], [0.6, 0.2, 0.2])
    bias_SR_indu = data_allrecord.bias_ret_indu.values / data_allrecord.bias_var_indu.values
    bias_score_indu = data_allrecord.bias_score_indu.values
    intercept_indu = data_allrecord.intercept_indu.values
    sc_indu = u_sc_cal([bias_SR_indu, bias_score_indu, intercept_indu], [0.6, 0.2, 0.2])
    sc_total = sc_mkt + sc_indu
    newcols = data_allrecord.columns.values.tolist() + ['SCORE']
    data_allrecord = data_allrecord.reindex(columns = newcols)
    for i, sc in zip(range(len(sc_total)), sc_total):
        data_allrecord.SCORE[i] = sc
    result = data_allrecord.sort_values('SCORE',ascending=False)
    return result

def select_Q2(data_allrecord, score_lvl = 0.80, beta_indu_lvl = 0.30, select_indu = 0):
    """
    风格：不偏
    行业：偏
    工具型基金
    行业：
    score_indu >= score_lvl
    某个beta_indu >= beta_indu_lvl
    筛选标准：
    1. bias ret/var
    2. bias score
    3. alpha
    """
    if select_indu != 0:
        # 指定了行业
        data_allrecord = data_allrecord.ix[data_allrecord.name_indu1 == select_indu]
    data_allrecord = data_allrecord.ix[(data_allrecord.beta_indu1 >= beta_indu_lvl) &
                                       (data_allrecord.score_indu >= score_lvl)]
    # select
    bias_SR_indu = data_allrecord.bias_ret_indu.values / data_allrecord.bias_var_indu.values
    bias_score_indu = data_allrecord.bias_score_indu.values
    intercept_indu = data_allrecord.intercept_indu.values
    sc_indu = u_sc_cal([bias_SR_indu, bias_score_indu, intercept_indu], [0.6, 0.2, 0.2])
    newcols = data_allrecord.columns.values.tolist() + ['SCORE']
    data_allrecord = data_allrecord.reindex(columns = newcols)
    for i, sc in zip(range(len(sc_indu)), sc_indu):
        data_allrecord.SCORE[i] = sc
    result = data_allrecord.sort_values('SCORE',ascending=False)
    return result

def select_Q3(data_allrecord, score_lvl = 0.60, betasum_lvl = 0.50, beta_mkt_lvl = 0.70, beta_indu_lvl = 0.40):
    """
    风格：不偏
    行业：不偏
    配置型基金
    筛选：
    score >= level & sum(beta_mkt) > betasum_lvl   剔除非股票基金
    beta <= level     不严重bias
    标准：
    intercept
    bias sharpe ratio
    """
    data_allrecord = data_allrecord.ix[(data_allrecord.beta_mkt1 + data_allrecord.beta_mkt2 + data_allrecord.beta_mkt3 >= betasum_lvl) &
                                       (data_allrecord.beta_mkt1 <= beta_mkt_lvl) &
                                       (data_allrecord.beta_indu1 <= beta_indu_lvl) &
                                       (data_allrecord.score_mkt >= score_lvl) &
                                       (data_allrecord.score_indu >= score_lvl)]
    # select
    bias_SR_mkt = data_allrecord.bias_ret_mkt.values / data_allrecord.bias_var_mkt.values
    intercept_mkt = data_allrecord.intercept_mkt.values
    sc_mkt = u_sc_cal([bias_SR_mkt, intercept_mkt], [0.3, 0.7])
    bias_SR_indu = data_allrecord.bias_ret_indu.values / data_allrecord.bias_var_indu.values
    intercept_indu = data_allrecord.intercept_indu.values
    sc_indu = u_sc_cal([bias_SR_indu, intercept_indu], [0.3, 0.7])
    sc_total = sc_mkt + sc_indu
    newcols = data_allrecord.columns.values.tolist() + ['SCORE']
    data_allrecord = data_allrecord.reindex(columns = newcols)
    for i, sc in zip(range(len(sc_total)), sc_total):
        data_allrecord.SCORE[i] = sc
    result = data_allrecord.sort_values('SCORE',ascending=False)
    return result

def select_Q4(data_allrecord, score_lvl = 0.80, beta_mkt_lvl = 0.60, select_mkt = 0, select_indu = 0):
    """
    风格：偏
    行业：不偏
    工具型基金
    风格：
    score_mkt >= score_lvl
    某个beta_mkt >= beta_mkt_lvl
    筛选标准：
    1. bias ret/var
    2. bias score
    3. alpha
    """
    if select_mkt != 0:
        # 指定了风格
        data_allrecord = data_allrecord.ix[data_allrecord.name_mkt1 == select_mkt]
    data_allrecord = data_allrecord.ix[(data_allrecord.beta_mkt1 >= beta_mkt_lvl) &
                                       (data_allrecord.score_mkt >= score_lvl)]
    # select
    bias_SR_mkt = data_allrecord.bias_ret_mkt.values / data_allrecord.bias_var_mkt.values
    bias_score_mkt = data_allrecord.bias_score_mkt.values
    intercept_mkt = data_allrecord.intercept_mkt.values
    sc_mkt = u_sc_cal([bias_SR_mkt, bias_score_mkt, intercept_mkt], [0.6, 0.2, 0.2])
    newcols = data_allrecord.columns.values.tolist() + ['SCORE']
    data_allrecord = data_allrecord.reindex(columns = newcols)
    for i, sc in zip(range(len(sc_mkt)), sc_mkt):
        data_allrecord.SCORE[i] = sc
    result = data_allrecord.sort_values('SCORE',ascending=False)
    return result

def u_sc_cal(data, factor):
    """
    to calc score for Manager
    data = [data1, data2, data3 ...]
    calc z score on each list of data
    weighted by factor
    all added up as score
    """
    result = np.zeros(len(data[0]))
    for dd,ff in zip(data,factor):
        result += ff * (dd - dd.mean()) / dd.std()
    return result


if __name__ == '__main__':
    main()
