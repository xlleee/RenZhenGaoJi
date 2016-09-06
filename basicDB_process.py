# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:19:41 2016

@author: lixiaolong
"""

import pyodbc
import pandas as pd
import numpy as np
import datetime


def process():
    """
    basic DB
    translated from SAS code
    """
    # date
    print(str(datetime.datetime.now()) + ': START')
    startdatestr = '2015-12-01'
    enddatestr = '2016-08-19'
    # JYDB db
    cnxn_jydb = pyodbc.connect("""
        DRIVER={SQL Server};
        SERVER=172.16.7.229;
        DATABASE=jydb;
        UID=sa;
        PWD=sa123456""")

    # let's go!
    ########################
    # FundTypeCode 晨星分类 jydb.CT_SystemConst LB=1273;
    # 1103-混合型 1105-债券型  1107-保本型  1109-货币型 1110-QDII
    # 1199-其他型 1101-股票型
    ########################
    # TYPE 1-契约型封闭  2-开放式  3-LOF 4-ETF 5-FOF
    # 6-创新型封闭式 7-开放式（带固定封闭期) 8-ETF联接 9-半开放式
    ########################
    # InvestmentType 投资类型（jydb.CT_SystemConst LB=1094）
    # 1.积极成长型 2.稳健成长性 3.中小企业成长型 4，平衡型
    # 5.资产重组型 6.科技型 7.指数型 8.优化指数型
    # 9.价值型 10.债券型 11.收益型 15.现金型
    # 20.内需增长型 99.综合型 21.生命周期型
    str_fundlist_codeonly = """
    select [InnerCode] from [JYDB].[dbo].[MF_FundArchives]
    where [Type] in (2,3,4) and [FundTypeCode] not in (1109,1105,1107,1199,1110)
    """
    str_fundlist_all = """
    select [InnerCode],[Type],[InvestmentType],[InvestStyle],[FundTypeCode]
    from [JYDB].[dbo].[MF_FundArchives]
    where [Type] in (2,3,4) and [FundTypeCode] not in (1109,1105,1107,1199,1110)
    """
    str_fundmanager = """
    select [InnerCode],[Name],[EducationLevel],[PracticeDate],
    [AccessionDate],[DimissionDate] from [JYDB].[dbo].[MF_FundManager]
    where [PostName] = 1
    and (DimissionDate is NULL or DimissionDate >=
    '""" + startdatestr + """ ')and AccessionDate<=
    '""" + enddatestr + """ '
    """
    str_fundnav_simple = """
    select [InnerCode],[EndDate],[UnitNV]
    from [JYDB].[dbo].[MF_NetValue] where [EndDate] between
    '""" + startdatestr + """' and '""" + enddatestr + """'
    and [InnerCode] in (""" + str_fundlist_codeonly + """)"""
    str_secumain = """
    select [InnerCode],[SecuCode],[SecuAbbr]
    from [JYDB].[dbo].[SecuMain]
    where [InnerCode] in (""" + str_fundlist_codeonly + """)"""
    str_adjustfactor = """
    select [InnerCode],[ExDiviDate],[GrowthRateFactor]
    from [JYDB].[dbo].[MF_AdjustingFactor]
    where [InnerCode] in (""" + str_fundlist_codeonly + """)"""
    str_investadvisor = """
    select A.[InnerCode], A.[InvestAdvisorCode], B.[InvestAdvisorAbbrName]
    from [JYDB].[dbo].[MF_FundArchives] A
    left join [JYDB].[dbo].[MF_InvestAdvisorOutline] B
    on A.[InvestAdvisorCode] = B.[InvestAdvisorCode]
    """
    print(str(datetime.datetime.now()) + ': READ DB')
    # df_fundlist_all ##############  1st df
    df_fundlist_all = pd.read_sql(str_fundlist_all, cnxn_jydb)
    # df_secumain ##################  2nd df
    df_secumain = pd.read_sql(str_secumain, cnxn_jydb)
    # df_fundnav_simple ############  3rd df
    df_fundnav_simple = pd.read_sql(str_fundnav_simple, cnxn_jydb)
    # df_adjustfactor ##############  4th df
    df_adjustfactor = pd.read_sql(str_adjustfactor, cnxn_jydb)
    # df_investadvisor #############  5th df
    df_investadvisor = pd.read_sql(str_investadvisor, cnxn_jydb)
    # df_fundmanager ###############  6th df
    df_fundmanager = pd.read_sql(str_fundmanager, cnxn_jydb)
    # 1.各种join
    print(str(datetime.datetime.now()) + ': MERGE DATA')
    # join secumain
    df_fundnav_all = pd.merge(df_fundnav_simple,
                              df_secumain,
                              on='InnerCode',
                              how='left')
    # join fundlist
    df_fundnav_all = pd.merge(df_fundnav_all,
                              df_fundlist_all,
                              on='InnerCode',
                              how='left')
    # join adjustfactor
    # left join on two columns!
    df_adjustfactor = df_adjustfactor.rename(columns={'ExDiviDate': 'EndDate'})
    df_fundnav_all = pd.merge(df_fundnav_all,
                              df_adjustfactor,
                              on=['InnerCode', 'EndDate'],
                              how='left')
    # join investadvisor
    df_fundnav_all = pd.merge(df_fundnav_all,
                              df_investadvisor,
                              on='InnerCode',
                              how='left')
    # sort
    df_fundnav_all = df_fundnav_all.sort_values(['InnerCode', 'EndDate'])
    df_fundnav_all['new index'] = range(len(df_fundnav_all))
    df_fundnav_all.set_index(keys=['new index'], drop=True, inplace=True)
    # reindex
    newcols = df_fundnav_all.columns.values.tolist() + \
        ['ID', 'dailyreturn', 'FundsofManager', 'ManagersofFund']
    df_fundnav_new = df_fundnav_all.reindex(columns=newcols)
    # 2.刷GrowthRateFactor
    print(str(datetime.datetime.now()) + ': CALCULATE GRATE')
    # 还是需要用到df_adjustfactor
    # df_adjustfactor包含了fundlist里所有基金所有日期的调整因子
    # 在当前的fundnav数据中，第一行如果为nan，则需要按照日期回溯adjustfactor
    # 得到在当前时间段之外，但需要沿用的调整因子
    for i in range(len(df_fundnav_new)):
        if np.isnan(df_fundnav_new.loc[i, 'GrowthRateFactor']):
            if i == 0 or df_fundnav_new.loc[i, 'InnerCode'] != df_fundnav_new.loc[i - 1, 'InnerCode']:
                # need to look up in df_adjustfactor
                df_temp = df_adjustfactor.ix[df_adjustfactor['InnerCode'] ==
                                             df_fundnav_new.loc[i, 'InnerCode']]
                df_temp = df_temp.sort_values('EndDate')
                df_temp = df_temp.ix[df_temp['EndDate'] <
                                     df_fundnav_new.loc[i, 'EndDate']]
                if len(df_temp) == 0:
                    # 空的！
                    temp = 1
                else:
                    temp = df_temp['GrowthRateFactor'].iloc[len(df_temp) - 1]
                df_fundnav_new.loc[i, 'GrowthRateFactor'] = temp
            else:
                df_fundnav_new.loc[i, 'GrowthRateFactor'] = \
                    df_fundnav_new.loc[i - 1, 'GrowthRateFactor']
    # 3.刷ManagerID
    print(str(datetime.datetime.now()) + ': MID')
    # if index fund: SecuAbbr + InnerCode
    # if not index fund:Name + PracticeDate
    newcols = df_fundmanager.columns.values.tolist() + ['ManagerID']
    df_fundmanager = df_fundmanager.reindex(columns=newcols)
    for i in range(len(df_fundmanager)):
        iCode = df_fundmanager.loc[i, 'InnerCode']
        # 有可能在fundlist里面找不到对应的，那么之后也不会merge进去
        if len(df_fundlist_all.ix[df_fundlist_all['InnerCode'] == iCode]) > 0:
            ivstmtType = df_fundlist_all.ix[df_fundlist_all[
                'InnerCode'] == iCode]['InvestmentType'].values[0]
            if ivstmtType in (7, 8):
                # index fund
                str_secuabbr = df_secumain.ix[df_secumain[
                    'InnerCode'] == iCode]['SecuAbbr'].values[0]
                mID = 'index' + str_secuabbr + str(iCode)
                # clean data in df_fundmanager
                df_fundmanager.loc[i, 'Name'] = str_secuabbr
                df_fundmanager.loc[i, 'EducationLevel'] = 0
                df_fundmanager.loc[i, 'PracticeDate'] = pd.tslib.NaT
                df_fundmanager.loc[i, 'AccessionDate'] = pd.tslib.NaT
                df_fundmanager.loc[i, 'DimissionDate'] = pd.tslib.NaT
            else:
                # not index fund
                str_name = df_fundmanager.loc[i, 'Name']
                if type(str_name) == float:
                    mID = 'empty'
                else:
                    if pd.isnull(df_fundmanager.loc[i, 'PracticeDate']):
                        # date == NaT
                        str_time = '00000000'
                    else:
                        str_time = df_fundmanager.loc[
                            i, 'PracticeDate'].strftime('%Y%m%d')
                    mID = str_name + str_time
            df_fundmanager.loc[i, 'ManagerID'] = mID
        else:
            df_fundmanager.loc[i, 'ManagerID'] = 'empty'
    dplcted = df_fundmanager.duplicated(subset=['InnerCode', 'Name'])
    df_fundmanager = df_fundmanager.ix[~dplcted]
    # reindex
    df_fundmanager['new index'] = range(len(df_fundmanager))
    df_fundmanager.set_index(keys=['new index'], drop=True, inplace=True)
    # 4.算dailyreturn
    print(str(datetime.datetime.now()) + ': RET')
    # 由于第一个位置的nv没有，无法算ret，所以每次读数据要重复一些，然后踢掉一段
    for i in range(len(df_fundnav_new)):
        if i == 0 or df_fundnav_new.loc[i, 'InnerCode'] != df_fundnav_new.loc[i - 1, 'InnerCode']:
            continue
        else:
            # calc daily ret
            UnitNVAdj = df_fundnav_new.loc[
                i, 'GrowthRateFactor'] * df_fundnav_new.loc[i, 'UnitNV']
            LastNVAdj = df_fundnav_new.loc[
                i - 1, 'GrowthRateFactor'] * df_fundnav_new.loc[i - 1, 'UnitNV']
            dailyreturn = UnitNVAdj / LastNVAdj - 1
            df_fundnav_new.loc[i, 'dailyreturn'] = dailyreturn
    # 5.刷ID
    print(str(datetime.datetime.now()) + ': ID')
    # merge FundManager
    df_fundnav_new = pd.merge(df_fundnav_new,
                              df_fundmanager,
                              on='InnerCode',
                              how='inner')
    # ID
    eDatearr = df_fundnav_new['EndDate'].values
    aDatearr = df_fundnav_new['AccessionDate'].values
    dDatearr = df_fundnav_new['DimissionDate'].values
    # xjb shua
    for i in range(len(dDatearr)):
        if dDatearr[i] is None:
            dDatearr[i] = pd.tslib.NaT
    df_fundnav_new.loc[:, 'DimissionDate'] = dDatearr
    # xjb shua wan
    mIDarr = df_fundnav_new['ManagerID'].values
    iCodearr = df_fundnav_new['InnerCode'].values
    idarr = list()
    for i in range(len(df_fundnav_new)):
        eDate = eDatearr[i]
        aDate = aDatearr[i]
        dDate = dDatearr[i]
        if pd.isnull(dDate):
            is_remain = eDate >= aDate
        else:
            is_remain = eDate >= aDate and eDate < np.datetime64(dDate)
        # is_remain = (eDate>=aDate) and (eDate<dDate or pd.isnull(dDate))
        is_index = mIDarr[i][0:5] == 'index'
        if is_remain or is_index:
            str_mID = mIDarr[i]
            str_eDate = str(eDate)[0:4] + str(eDate)[5:7] + str(eDate)[8:10]
            str_iCode = str(iCodearr[i])
            str_ID = str_mID + 'D' + str_eDate + 'F' + str_iCode
            idarr.append(str_ID)
        else:
            idarr.append('empty')
    df_fundnav_new.loc[:, 'ID'] = idarr
    # clear empty
    df_fundnav_fine = df_fundnav_new.ix[df_fundnav_new['ID'] != 'empty']
    # if duplicated ID
    dplcted = df_fundnav_fine.duplicated(subset=['ID'])
    df_fundnav_fine = df_fundnav_fine.ix[~dplcted]
    # reindex
    df_fundnav_fine['new index'] = range(len(df_fundnav_fine))
    df_fundnav_fine.set_index(keys=['new index'], drop=True, inplace=True)
    # 6.算FundsofManager和ManagersofFund
    print(str(datetime.datetime.now()) + ': FM & MF')
    # 如果把整个df拿出来比较，会产生大量冗余的计算
    # print(datetime.datetime.now())
    # using np.ndarray instead of DataFrame
    eDateArr = df_fundnav_fine['EndDate'].values
    iCodeArr = df_fundnav_fine['InnerCode'].values
    mIDArr = df_fundnav_fine['ManagerID'].values
    fsOFmng = np.zeros(len(df_fundnav_fine))
    mngOFfs = np.zeros(len(df_fundnav_fine))
    # go
    lastEndDate = eDateArr[0]
    iCodeArr_temp = iCodeArr[eDateArr == lastEndDate]
    mIDArr_temp = mIDArr[eDateArr == lastEndDate]
    for i in range(len(df_fundnav_fine)):
        eDate = eDateArr[i]
        mID = mIDArr[i]
        iCode = iCodeArr[i]
        if eDate != lastEndDate:
            lastEndDate = eDate
            iCodeArr_temp = iCodeArr[eDateArr == lastEndDate]
            mIDArr_temp = mIDArr[eDateArr == lastEndDate]
        sameM = len(iCodeArr_temp[mIDArr_temp == mID])
        sameF = len(mIDArr_temp[iCodeArr_temp == iCode])
        fsOFmng[i] = sameM
        mngOFfs[i] = sameF
        # print(str(i) + ',' + str(sameM) + ',' + str(sameF))
    df_fundnav_fine.loc[:, 'FundsofManager'] = fsOFmng
    df_fundnav_fine.loc[:, 'ManagersofFund'] = mngOFfs
    return df_fundnav_fine
    # print(datetime.datetime.now())


def write_sql(df_fundnav_fine, is_create):
    # 7.导入数据库
    print(str(datetime.datetime.now()) + ': WRITE DB')
    # JRGCB db
    cnxn_jrgcb = pyodbc.connect("""
        DRIVER={SQL Server};
        SERVER=172.16.7.166;
        DATABASE=jrgcb;
        UID=sa;
        PWD=sa123456""")
    writestartdatestr = '2016-01-01'
    writestartdate = datetime.datetime.strptime(writestartdatestr, '%Y-%m-%d')
    cursor_jrgcb = cnxn_jrgcb.cursor()
    if is_create:
        # create new table
        str_create = """
        CREATE TABLE [jrgcb].[dbo].[FundAndManagerData_v2]
        (InnerCode INT NULL,
        EndDate DATETIME NULL,
        UnitNV  FLOAT(31) NULL,
        SecuCode VARCHAR(255) NULL,
        SecuAbbr VARCHAR(255) NULL,
        Type INT NULL,
        InvestmentType INT NULL,
        InvestStyle INT NULL,
        FundTypeCode INT NULL,
        GrowthRateFactor FLOAT(31) NULL,
        InvestAdvisorCode INT NULL,
        InvestAdvisorAbbrName VARCHAR(255) NULL,
        ID VARCHAR(255) NOT NULL,
        dailyreturn FLOAT(31) NULL,
        FundsofManager INT NULL,
        ManagersofFund INT NULL,
        Name VARCHAR(255) NULL,
        EducationLevel INT NULL,
        PracticeDate DATETIME NULL,
        AccessionDate DATETIME NULL,
        DimissionDate DATETIME NULL,
        ManagerID VARCHAR(255) NULL,
        PRIMARY KEY (ID))
        """
        cursor_jrgcb.execute(str_create)
        cursor_jrgcb.commit()
    # write lines
    for row in df_fundnav_fine.values.tolist():
        if row[1] >= writestartdate:
            str_smart_sql, value_list = smart_write_sql(df_fundnav_fine.columns.values.tolist(),
                                                        row,
                                                        '[jrgcb].[dbo].[FundAndManagerData_v2]')
            cursor_jrgcb.execute(str_smart_sql, value_list)
            cursor_jrgcb.commit()
    print(str(datetime.datetime.now()) + ': END')


def smart_write_sql(cols, row, db_name):
    """
    能够判断row里面那些为nan or nat
    然后在sql里面不要导入
    """
    str_temp = 'INSERT INTO ' + db_name
    str_col = '('
    qMark_counts = 0
    value_list = []
    for i in range(len(row)):
        isRecord = {int: lambda x: not np.isnan(x),
                    float: lambda x: not np.isnan(x),
                    pd.tslib.Timestamp: lambda x: not pd.isnull(x),
                    pd.tslib.NaTType: lambda x: False,
                    str: lambda x: True}[type(row[i])](row[i])
        if isRecord:
            # 这项输入sql
            str_col += '[' + cols[i] + '],'
            qMark_counts += 1
            value_list.append(row[i])
    str_col = str_col[0:-1] + ')'
    qMarks = '(' + ('?,' * qMark_counts)[0:-1] + ')'
    str_smart_sql = str_temp + str_col + 'VALUES' + qMarks
    return str_smart_sql, value_list
