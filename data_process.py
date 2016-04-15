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









def organize_data(ManagerID, cnxn_jydb, cnxn_jrgcb):
    sql_allrecord = """
        SELECT [InnerCode], [EndDate], [dailyreturn], [FundsofManager], [ManagersofFund]
        FROM [jrgcb].[dbo].[FundAndManagerData]
        WHERE [ManagerID] = """ + ManagerID + """
        ORDER BY [EndDate]
        """
    
