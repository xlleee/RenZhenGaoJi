# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:22:09 2016

@author: lixiaolong
"""

import pyodbc
import pandas as pd
import numpy as np

def main():
    """
    select by input:
    用dict作为输入，满分 = 10
    input 1: mkt
    input 2: indu
    input 3: alpha beta
    """
    # date
    date_str = '2016-05-30'
    # input
    input_mkt = {'沪深300指数': 6,
                 '中证小盘500指数': 6,
                 '中证1000指数': 6}
    input_indu = {'中信证券-房地产': 6,
                  '中信证券-国防军工': 6}
    input_prefer = {'bias mkt': 6,
                  'alpha mkt': 6,
                  'bias indu': 6,
                  'alpha indu': 6}
    # sql
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
    data_allrecord = pd.read_sql(select_data_sql,
                                 cnxn_jrgcb,
                                 index_col = 'ManagerID')
    # wash1: 剔除非股票基金
    data_allrecord = wash1_non_equity(data_allrecord)
    # 打分：根据input prefer打分，然后排序，再返回
    data_allrecord = rate_and_rank(data_allrecord, input_prefer)
    # wash2: 剔除不包含要求行业的基金
    data_allrecord = wash2_non_indu(data_allrecord, input_indu)
    # 选鸡


def select_chicken(data_allrecord, input_mkt, input_indu):
    """
    选基流程：
    1. 确定样本池大小
    2. 取一个大的范围N1，比如说样本池的一半，且不要小于20
        N1个基金中要包含所选的mkt和indu
        if not，扩大，如果扩不大了，就报错！
    3. 对N1优化权重
    4. 评价优化结果，踢掉权重小于某阈值的基金，然后重复，N2 = N1 - 踢掉
    5. 直到踢不掉任何基金
    """
    temp_size = 20
    sample_size = max([int(0.5 * len(data_allrecord)), temp_size])
    # 1. check indu
    data_sample = data_allrecord.iloc[0:sample_size]
    lgt = False
    list_of_indu = list(input_indu.keys())
    while not lgt:
        lgt = True
        for indu in list_of_indu:
            # 三个行业里面包含了indu
            templgt = ((data_sample.name_indu1 == indu).any() or
                       (data_sample.name_indu2 == indu).any() or
                       (data_sample.name_indu3 == indu).any())
            # 在indu list里面取 and
            lgt = lgt and templgt
        # 如果发现并找不到indu，则扩大范围
        if not lgt:
            if sample_size < len(data_allrecord):
                sample_size = min([len(data_allrecord), sample_size + 10])
                data_sample = data_allrecord.iloc[0:sample_size]
            else:
                # 已经看完了所有基金，还是没有indu
                print('No indu included! Check your data!')
                return data_sample
    # 2. optimize weight on data_sample
    # get mkt and indu beta
    beta_mkt, beta_indu = get_beta(data_sample)
    # call opt








def opt_func(x, beta_mkt, beta_indu,
             input_mkt, input_indu,
             alpha_mkt, alpha_indu, alpha_term):
    """
    Object function
    """
    # mkt terms


    # indu terms



    # alpha terms







def opt_func_deriv(x, beta_mkt, beta_indu,
                   input_mkt, input_indu,
                   alpha_mkt, alpha_indu, alpha_term):
    """
    Derivative of object function
    """

def get_beta(data_sample):
    """
    split mkt and indu beta
    """
    # mkt def
    mktname_unq = np.append(data_sample.name_mkt1.unique(),
                            data_sample.name_mkt2.unique())
    mktname_unq = np.append(mktname_unq,data_sample.name_mkt3.unique())
    mktname_unq = np.unique(mktname_unq)
    beta_mkt = pd.DataFrame(columns = mktname_unq, index = data_sample.index.values)
    # indu def
    induname_unq = np.append(data_sample.name_indu1.unique(),
                            data_sample.name_indu2.unique())
    induname_unq = np.append(induname_unq,data_sample.name_indu3.unique())
    induname_unq = np.unique(induname_unq)
    beta_indu = pd.DataFrame(columns = induname_unq, index = data_sample.index.values)
    for index,row in data_sample.iterrows():
        beta_mkt.ix[index, row.name_mkt1] = row.beta_mkt1
        beta_mkt.ix[index, row.name_mkt2] = row.beta_mkt2
        beta_mkt.ix[index, row.name_mkt3] = row.beta_mkt3
        beta_indu.ix[index, row.name_indu1] = row.beta_indu1
        beta_indu.ix[index, row.name_indu2] = row.beta_indu2
        beta_indu.ix[index, row.name_indu3] = row.beta_indu3
    beta_mkt = beta_mkt.fillna(value = 0)
    beta_indu = beta_indu.fillna(value = 0)
    return beta_mkt, beta_indu

def wash1_non_equity(data_allrecord, betasum_lvl = 0.50, score_lvl = 0.60):
    """
    score >= level & sum(beta_mkt) > betasum_lvl   剔除非股票基金
    """
    data_allrecord = data_allrecord.ix[(data_allrecord.beta_mkt1 +
                                        data_allrecord.beta_mkt2 +
                                        data_allrecord.beta_mkt3 >=
                                        betasum_lvl) &
                                       (data_allrecord.score_mkt >= score_lvl)]
    return data_allrecord

def wash2_non_indu(data_allrecord, input_indu):
    """
    剔除不包含行业的
    """
    list_of_indu = list(input_indu.keys())
    lgt = np.array([False] * len(data_allrecord.name_indu1.values))
    for indu in list_of_indu:
        lgt = lgt | (data_allrecord.name_indu1.values == indu)
        lgt = lgt | (data_allrecord.name_indu2.values == indu)
        lgt = lgt | (data_allrecord.name_indu3.values == indu)
    return data_allrecord.ix[lgt]

def rate_and_rank(data_allrecord, input_prefer):
    """
    1. rate：根据prefer进行打分
    2. rank：排序
    打分算法：
    首先分mkt和indu，然后对于mkt和indu：
    bias prefer × bias项 + alpha prefer × alpha项
    bias项：衡量bias ret的质量
    alpha项：衡量alpha的质量
    所以就是四项的加权平均
    """
    # 四个数据
    bias_SR_mkt = data_allrecord.bias_ret_mkt.values / data_allrecord.bias_var_mkt.values
    bias_SR_indu = data_allrecord.bias_ret_indu.values / data_allrecord.bias_var_indu.values
    alpha_mkt = data_allrecord.intercept_mkt.values
    alpha_indu = data_allrecord.intercept_indu.values
    # 算score
    normalize = 1 / sum(list(input_prefer.values()))
    score = (input_prefer['bias mkt'] * (bias_SR_mkt - bias_SR_mkt.mean()) / bias_SR_mkt.std() +
             input_prefer['bias indu'] * (bias_SR_indu - bias_SR_indu.mean()) / bias_SR_indu.std() +
             input_prefer['alpha mkt'] * (alpha_mkt - alpha_mkt.mean()) / alpha_mkt.std() +
             input_prefer['alpha indu'] * (alpha_indu - alpha_indu.mean()) / alpha_indu.std()) * normalize
    newcols = data_allrecord.columns.values.tolist() + ['SCORE']
    data_allrecord = data_allrecord.reindex(columns = newcols)
    for i, sc in zip(range(len(score)), score):
        data_allrecord.SCORE[i] = sc
    data_allrecord = data_allrecord.sort_values('SCORE',ascending=False)
    return data_allrecord


if __name__ == '__main__':
    main()
