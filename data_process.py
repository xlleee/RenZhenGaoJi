# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:04:38 2016

@author: lixiaolong
"""

import pyodbc
import pandas as pd
import numpy as np
from sklearn import linear_model

########################################################
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
# sql to select distinct fund manager
sql_allmng = """
    SELECT DISTINCT [ManagerID]
    FROM [jrgcb].[dbo].[FundAndManagerData]
    ORDER BY [ManagerID]
    """
data_allmng = pd.read_sql(sql_allmng, cnxn_jrgcb)
########################################################
# read index data for JYDB
sql_betaindex = """
    SELECT A.SecuCode, B.TradingDay, B.ChangePCT, A.ChiNameAbbr, A.ChiName
    FROM [JYDB].[dbo].[SecuMain] A, [JYDB].[dbo].[QT_IndexQuote] B
    WHERE A.InnerCode = B.InnerCode AND A.SecuCode in ('000300','000905','000852')
    ORDER BY A.SecuCode, B.TradingDay"""
data_betaindex = pd.read_sql(sql_betaindex, cnxn_jydb)
sql_induindex = """
    SELECT A.SecuCode, B.TradingDay, B.ChangePCT, A.ChiNameAbbr, A.ChiName, A.InnerCode
    FROM [JYDB].[dbo].[SecuMain] A, [JYDB].[dbo].[QT_IndexQuote] B
    WHERE A.InnerCode = B.InnerCode AND A.SecuCode in
    ('CI005001','CI005002','CI005003','CI005004','CI005005',
    'CI005006','CI005007','CI005008','CI005009','CI005010',
    'CI005011','CI005012','CI005013','CI005014','CI005015',
    'CI005016','CI005017','CI005018','CI005019','CI005020',
    'CI005021','CI005022','CI005023','CI005024','CI005025',
    'CI005026','CI005027','CI005028','CI005029')
    ORDER BY A.SecuCode, B.TradingDay"""
data_induindex = pd.read_sql(sql_induindex, cnxn_jydb)







def organize_data(ManagerID, cnxn_jydb, cnxn_jrgcb, data_betaindex, data_induindex):
    # read mng record data
    sql_allrecord = """
        SELECT [InnerCode], [EndDate], [dailyreturn], [FundsofManager], [ManagersofFund]
        FROM [jrgcb].[dbo].[FundAndManagerData]
        WHERE [ManagerID] = """ + ManagerID + """
        ORDER BY [EndDate]
        """
    data_allrecord = read_sql(sql_allrecord, cnxn_jrgcb)
