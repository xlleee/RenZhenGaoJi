# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:22:09 2016

@author: lixiaolong
"""

import pyodbc
import pandas as pd
import numpy as np
from scipy.optimize import minimize

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
    raw_data = pd.read_sql(select_data_sql,
                                 cnxn_jrgcb,
                                 index_col = 'ManagerID')
    # input
    input_mkt = {'沪深300指数': 3.,
                 '中证小盘500指数': 4.,
                 '中证1000指数': 5.}
    input_indu = {'中信证券-计算机': 4.,
                  '中信证券-食品饮料': 5.,
                  '中信证券-医药': 6.}
    input_prefer = {'bias mkt': 3.,
                  'alpha mkt': 3.,
                  'bias indu': 4.,
                  'alpha indu': 6.}
    # wash1: 剔除非股票基金
    data_allrecord = wash1_non_equity(raw_data)
    # 打分：根据input prefer打分，然后排序，再返回
    data_allrecord = rate_and_rank(data_allrecord, input_prefer)
    # wash2: 剔除不包含要求行业的基金
    data_allrecord = wash2_non_indu(data_allrecord, input_indu)
    # 选鸡
    # 参数很关键！
    input_parameters = {'indu_con_factor': .30,    # input行业集中度参数，表示几个行业的beta之和
                        'score_factor': .000001,   # 目标函数中score项的系数，越大越重要
                        'mkt_factor': 1.,          # mkt项
                        'indu_factor': 1.5,        # indu项
                        'cut_threshold':.01}     # cut low weight的时候的门槛，特别关键！
    ww, mkt_exp, indu_exp, score_exp, data_sample = select_chicken(data_allrecord,
                                                                   input_mkt,
                                                                   input_indu,
                                                                   input_parameters)

def select_chicken(data_allrecord, input_mkt, input_indu, input_parameters):
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
    is_continue = True
    while is_continue:
        # get mkt and indu beta
        beta_mkt, beta_indu = get_beta(data_sample)
        # call opt
        # parameters
        ini = [1 / len(beta_mkt)] * len(beta_mkt)
        wgt_bounds = [(0,1)] * len(beta_mkt)
        # run opt
        cons = ({'type':'eq',
                 'fun':lambda x: np.array(sum(x) - 1),
                 'jac':lambda x: np.array([1] * len(x))},
                {'type':'ineq',
                 'fun':con_indu_beta_sum,
                 'jac':con_indu_beta_sum_deriv,
                 'args':(beta_indu, input_indu, input_parameters['indu_con_factor'])})
        res = minimize(opt_func,
                       ini,
                       args=(beta_mkt,beta_indu,input_mkt,
                             input_indu,data_sample.SCORE.values,
                             input_parameters['score_factor'],
                             input_parameters['mkt_factor'],
                             input_parameters['indu_factor']),
                       jac=opt_func_deriv,
                       constraints=cons,
                       bounds=wgt_bounds,
                       method='SLSQP',
                       options={'disp':True})
        ww = res.x
        mkt_exp, indu_exp, score_exp = weight_summary(ww,
                                                      beta_mkt,
                                                      beta_indu,
                                                      data_sample.SCORE.values)
        # wash3: get rid of weight ~0 funds
        is_cut, data_sample = cut_low_weight(ww, data_sample, input_indu, input_parameters['cut_threshold'])
        if not is_cut:
            # already minimun
            is_continue = False
    # 3. return
    return ww, mkt_exp, indu_exp, score_exp, data_sample





def opt_func(x, beta_mkt, beta_indu,
             input_mkt, input_indu,
             score, score_factor,
             mkt_factor, indu_factor):
    """
    Object function
    x is the weight on each fund
    目标是使在各个mkt和indu上的beta总暴露达到某个比例：input_mkt, input_indu
    同时对整体score有要求，越高越好，score_factor用来调整敏感程度
    优化目标为min
    1. mkt term:
    B_i = sum(weight * beta_mkt_i)   ----- total exposure on mkt_i
    Bk_i = B_i / input_mkt[i]         ----- weighted total exposure
    mkt term = (Bk_1 - Bk_2)**2 + ... + (Bk_n-1 - Bk_n)**2 + (Bk_n - Bk_1)**2
    2. indu term: similar with mkt term
    3. score term: sum(weight * score) * factor
    total = mkt term + indu term - score term
    """
    # mkt terms
    Bk_mkt = list()
    for k in iter(input_mkt):
        temp_b = sum(x * beta_mkt[k].values) / input_mkt[k]
        Bk_mkt.append(temp_b)
    mkt_term = circle_diff_power_sum(Bk_mkt) * mkt_factor
    # indu terms
    Bk_indu = list()
    for k in iter(input_indu):
        temp_b = sum(x * beta_indu[k].values) / input_indu[k]
        Bk_indu.append(temp_b)
    indu_term = circle_diff_power_sum(Bk_indu) * indu_factor
    # alpha terms
    score_term = sum(x * score) * score_factor
    # return: mkt + indu - score
    return mkt_term + indu_term - score_term

def opt_func_deriv(x, beta_mkt, beta_indu,
                   input_mkt, input_indu,
                   score, score_factor,
                   mkt_factor, indu_factor):
    """
    Derivative of object function
    给minimize函数的jacobi，一列数，对应x的个数：list of dfdx
    dfdx_i = d(opt_func)/d(x_i)
    mkt和indu项：括号的平方求导，每个括号里都有x_i
    """
    list_of_dfdx = list()
    # beta of mkt
    Bk_mkt = list()
    for k in iter(input_mkt):
        temp_b = sum(x * beta_mkt[k].values) / input_mkt[k]
        Bk_mkt.append(temp_b)
    # beta of indu
    Bk_indu = list()
    for k in iter(input_indu):
        temp_b = sum(x * beta_indu[k].values) / input_indu[k]
        Bk_indu.append(temp_b)
    # loop through x
    for i in range(len(x)):
        # mkt
        betak_mkt = list()
        for k in iter(input_mkt):
            temp_bk = beta_mkt[k].values[i] / input_mkt[k]
            betak_mkt.append(temp_bk)
        mkt_term = circle_deriv_sum(Bk_mkt, betak_mkt) * mkt_factor
        # indu
        betak_indu = list()
        for k in iter(input_indu):
            temp_bk = beta_indu[k].values[i] / input_indu[k]
            betak_indu.append(temp_bk)
        indu_term = circle_deriv_sum(Bk_indu, betak_indu) * indu_factor
        # score
        score_term = score_factor * score[i]
        # append dfdx_i
        list_of_dfdx.append(mkt_term + indu_term - score_term)
    # return np.array
    return np.array(list_of_dfdx)

def con_indu_beta_sum(x, beta_indu, input_indu, level):
    """
    for con
    beta exp on input_indu should >= level
    """
    beta_sum = 0
    for k in iter(input_indu):
        beta_sum += sum(beta_indu[k].values * x)
    return beta_sum - level

def con_indu_beta_sum_deriv(x, beta_indu, input_indu, level):
    """
    deriv of con
    array of x_i
    """
    list_of_dfdx = list()
    for i in range(len(x)):
        temp = 0
        for k in iter(input_indu):
            temp += x[i] * beta_indu[k].values[i]
        list_of_dfdx.append(temp)
    return np.array(list_of_dfdx)

def weight_summary(x, beta_mkt, beta_indu, score):
    """
    输出x对应的各种暴露
    """
    # mkt
    mkt_exp = dict()
    for i in range(len(beta_mkt.columns)):
        temp_exp = sum(x * beta_mkt.iloc[:,i].values)
        mkt_exp.update({beta_mkt.columns[i]:temp_exp})
    # indu
    indu_exp = dict()
    for i in range(len(beta_indu.columns)):
        temp_exp = sum(x * beta_indu.iloc[:,i].values)
        indu_exp.update({beta_indu.columns[i]:temp_exp})
    # score
    score_exp = sum(x * score)
    return mkt_exp, indu_exp, score_exp

def circle_deriv_sum(B, bk):
    """
    for opt_func_deriv obly
    """
    # term1
    temp = B[1:]
    temp.append(B[0])
    term1 = np.array(B) - np.array(temp)
    # term2
    temp = bk[1:]
    temp.append(bk[0])
    term2 = np.array(bk) - np.array(temp)
    return sum(2 * term1 * term2)

def circle_diff_power_sum(Bk, power = 2):
    """
    (x1 - x2)**p + (x2 - x3)**p + ... + (xn - x1)**p
    """
    temp = Bk[1:]
    temp.append(Bk[0])
    diff = np.array(Bk) - np.array(temp)
    return sum(diff ** power)

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

def cut_low_weight(ww, data_sample, input_indu, threshold):
    """
    cut low weight fund, and keep all input indu
    threshold from high to low
    greed is good
    """
    decay = 0.0001
    is_continue = True
    while is_continue:
        lgt = ww > threshold
        temp_data_df = data_sample.ix[lgt]
        # check indu names contained
        induname_unq = np.append(temp_data_df.name_indu1.unique(),
                                temp_data_df.name_indu2.unique())
        induname_unq = np.append(induname_unq,temp_data_df.name_indu3.unique())
        induname_unq = np.unique(induname_unq)
        is_contained = True
        for k in iter(input_indu):
            tt = k in induname_unq
            is_contained = is_contained and tt
        if is_contained:
            is_continue = False
            result = temp_data_df
        else:
            threshold -= decay
            if threshold <= 0:
                is_continue = False
                result = data_sample
    is_cut = len(data_sample) > len(result)
    return is_cut, result

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
