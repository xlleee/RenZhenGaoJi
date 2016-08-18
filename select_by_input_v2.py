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
    input 1: mkt
    input 2: indu
    input 3: prefer
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
    input_mkt = {'沪深300指数': 8.,
                 '中证小盘500指数': 1.,
                 '中证1000指数': 1.}
    input_indu = {'中信证券-计算机': 1.,
                  '中信证券-食品饮料': 5.,
                  '中信证券-医药': 5.}
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
    input_parameters = {'mkt_width': 1,
                        'indu_width': 1}
    ww, mkt_exp, indu_exp, score_exp, data_sample = select_chicken(data_allrecord,
                                                                   input_mkt,
                                                                   input_indu,
                                                                   input_parameters)

def backtest(ww, mkt_exp, indu_exp, score_exp, data_sample, input_mkt, input_indu, startdatestr, enddatestr, cnxn_jrgcb):
    """
    基于选出的基金组合进行回测
    1. 看alpha
    2. 看跟踪误差
    """
    # read data
    



def select_chicken(data_allrecord, input_mkt, input_indu, input_parameters):
    """
    选基流程：
    1. 根据INDU的input选取样本池，分数越高的行业对应的基金越多
    2. 优化：以SCORE最高为目标，约束为MKT和INDU的input分数比例
    """
    # 1. select pool
    # pool
    data_sample = pd.DataFrame(columns = data_allrecord.columns)
    # copy of indu_names, easy to loop
    # top 3 indu
    indu_names = data_allrecord[['name_indu1','name_indu2','name_indu3']]
    for k in iter(input_indu):
        num_of_fund = input_indu[k]
        num_selected = 0
        row_idx= 0
        indu_idx = 0 # 0,1,2
        while indu_idx <= 2:
            if indu_names.iloc[row_idx, indu_idx] == k:
                # then select
                data_sample = data_sample.append(data_allrecord.iloc[row_idx,:])
                num_selected += 1
                if num_selected >= num_of_fund:
                    # done
                    break
            # next row / indu
            row_idx += 1
            if row_idx >= len(data_allrecord):
                indu_idx += 1
                row_idx = 0
    # 2. optimize
    # get mkt and indu beta
    beta_mkt, beta_indu = get_beta(data_sample)
    # call opt
    # parameters
    ini = [1 / len(beta_mkt)] * len(beta_mkt)
    wgt_bounds = [(0.0,1.0)] * len(beta_mkt)
    # run opt
    is_go_on = True
    diff = calc_diff(ini, beta_mkt, beta_indu, input_mkt, input_indu, input_parameters)
    while is_go_on:
        # generate cons
        cons = list()
        # eq con
        cons.append({'type':'eq',
                     'fun':lambda x: np.array(sum(x) - 1),
                     'jac':lambda x: np.array([1] * len(x))})
        # mkt ineq
        for i,k in zip(range(len(input_mkt)),iter(input_mkt)):
            if i < len(input_mkt) - 1:
                condict = {'type':'ineq',
                           'fun':con_func,
                           'jac':con_jac,
                           'args':(beta_mkt[k].values,
                                   beta_mkt[list(input_mkt.keys())[i+1]].values,
                                   input_mkt[k],
                                   list(input_mkt.values())[i+1],
                                   input_parameters['mkt_width'])}
                cons.append(condict)
            else:
                # last item in input_mkt
                condict = {'type':'ineq',
                           'fun':con_func,
                           'jac':con_jac,
                           'args':(beta_mkt[k].values,
                                   beta_mkt[list(input_mkt.keys())[0]].values,
                                   input_mkt[k],
                                   list(input_mkt.values())[0],
                                   input_parameters['mkt_width'])}
                cons.append(condict)
        # indu ineq
        for i,k in zip(range(len(input_indu)),iter(input_indu)):
            if i < len(input_indu) - 1:
                condict = {'type':'ineq',
                           'fun':con_func,
                           'jac':con_jac,
                           'args':(beta_indu[k].values,
                                   beta_indu[list(input_indu.keys())[i+1]].values,
                                   input_indu[k],
                                   list(input_indu.values())[i+1],
                                   input_parameters['indu_width'])}
                cons.append(condict)
            else:
                # last item in input_indu
                condict = {'type':'ineq',
                           'fun':con_func,
                           'jac':con_jac,
                           'args':(beta_indu[k].values,
                                   beta_indu[list(input_indu.keys())[0]].values,
                                   input_indu[k],
                                   list(input_indu.values())[0],
                                   input_parameters['indu_width'])}
                cons.append(condict)
        # finish con, start opt
        res = minimize(opt_func,
                       ini,
                       args=(data_sample.SCORE.values),
                       jac=opt_func_deriv,
                       constraints=cons,
                       bounds=wgt_bounds,
                       method='SLSQP',
                       options={'disp':True})
        new_diff = calc_diff(res.x, beta_mkt, beta_indu, input_mkt, input_indu, input_parameters)
        if res.success:
            if new_diff <= diff * 0.9:
                diff = new_diff
                input_parameters['mkt_width'] = input_parameters['mkt_width'] * 0.8
                input_parameters['indu_width'] = input_parameters['indu_width'] * 0.8
            else:
                is_go_on = False
        else:
            input_parameters['mkt_width'] = input_parameters['mkt_width'] * 2
            input_parameters['indu_width'] = input_parameters['indu_width'] * 2
    ww = res.x
    mkt_exp, indu_exp, score_exp = weight_summary(ww,
                                                  beta_mkt,
                                                  beta_indu,
                                                  data_sample.SCORE.values)
    # 3. return
    return ww, mkt_exp, indu_exp, score_exp, data_sample

def calc_diff(x, beta_mkt, beta_indu, input_mkt, input_indu, input_parameters):
    """
    calc diff with input_mkt and input_indu
    """
    temp = 0
    for i,k in zip(range(len(input_mkt)),iter(input_mkt)):
        if i < len(input_mkt) - 1:
            temp += con_func(x,
                             beta_mkt[k].values,
                             beta_mkt[list(input_mkt.keys())[i+1]].values,
                             input_mkt[k],
                             list(input_mkt.values())[i+1],
                             input_parameters['mkt_width'])
        else:
            temp += con_func(x,
                             beta_mkt[k].values,
                             beta_mkt[list(input_mkt.keys())[0]].values,
                             input_mkt[k],
                             list(input_mkt.values())[0],
                             input_parameters['mkt_width'])
    for i,k in zip(range(len(input_indu)),iter(input_indu)):
        if i < len(input_indu) - 1:
            temp += con_func(x,
                             beta_indu[k].values,
                             beta_indu[list(input_indu.keys())[i+1]].values,
                             input_indu[k],
                             list(input_indu.values())[i+1],
                             input_parameters['indu_width'])
        else:
            temp += con_func(x,
                             beta_indu[k].values,
                             beta_indu[list(input_indu.keys())[0]].values,
                             input_indu[k],
                             list(input_indu.values())[0],
                             input_parameters['indu_width'])
    return temp

def con_func(x, beta1, beta2, k1, k2, width):
    return  width - (sum(beta1 * x) / k1 - sum(beta2 * x) / k2) ** 2 # >=0

def con_jac(x, beta1, beta2, k1, k2, width):
    list_of_dfdx = list()
    for i in range(len(x)):
        temp = -2 * (sum(beta1 * x) / k1 - sum(beta2 * x) / k2) * (beta1[i]/k1 - beta2[i]/k2)
        list_of_dfdx.append(temp)
    return np.array(list_of_dfdx)

def opt_func(x, score):
    """
    simply  - x * score
    """
    return -1 * sum(x * score)

def opt_func_deriv(x, score):
    """
    Derivative of object function
    """
    return -1 * score

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

def get_beta(data_sample):
    """
    split mkt and indu beta
    """
    # time
    date_array = data_sample.index.values
    # mkt
    mktname_list = list()
    for i in range(1,4):
        # from 1 to 3
        temp = eval('data_sample.loc[0,"name_mkt' + str(i) + '"]')
        mktname_list.append(temp)
    mkt_beta_df = pd.DataFrame(columns = mktname_list, index = date_array)
    # indu
    induname_list = list()
    for i in range(1,30):
        # from 1 to 29
        temp = eval('data_sample.loc[0,"name_indu' + str(i) + '"]')
        induname_list.append(temp)
    indu_beta_df = pd.DataFrame(columns = induname_list, index = date_array)
    for index,row in data_sample.iterrows():
        for i in range(1,4):
            eval('mkt_beta_df.ix[index, row.name_mkt' + str(i) + '] = row.beta_mkt' + str(i))
        for i in range(1,30):
            eval('indu_beta_df.ix[index, row.name_indu' + str(i) +'] = row.beta_indu' + str(i))
    mkt_beta_df = mkt_beta_df.fillna(value = 0)
    indu_beta_df = indu_beta_df.fillna(value = 0)
    return mkt_beta_df, indu_beta_df

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
