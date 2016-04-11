# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:02:46 2016

@author: lixiaolong
"""


import pyodbc
import pandas
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=172.16.7.229;DATABASE=jydb;UID=sa;PWD=sa123456')
cursor = cnxn.cursor()

# using pandas
# pretty good!
sql = """
    SELECT [SalesDepartmentName],[TradingDay],[SerialNum],[InnerCode] 
    FROM [JYDB].[dbo].[LC_7PercentChange] 
    WHERE [SalesDepartmentName] LIKE '%衢州%' AND 
    [TradingDay] > '2015-01-01'ORDER BY [TradingDay]
    """
data = pandas.read_sql(sql,cnxn)

# using LIKE
cursor.execute("""
    SELECT [SalesDepartmentName],[TradingDay],[SerialNum],[InnerCode] 
    FROM [JYDB].[dbo].[LC_7PercentChange] 
    WHERE [SalesDepartmentName] LIKE '%衢州%' AND 
    [TradingDay] > '2015-01-01'ORDER BY [TradingDay]
    """)  
    
# using alias
cursor.execute("""
    SELECT main.[SecuCode], main.[SecuAbbr], 
    a.[SalesDepartmentName], a.[TradingDay], a.[SerialNum], a.[InnerCode] 
    FROM [JYDB].[dbo].[SecuMain] as main, [JYDB].[dbo].[LC_7PercentChange] as a 
    WHERE main.[InnerCode]=a.[InnerCode]
    AND a.[SalesDepartmentName] LIKE '%衢州%' AND 
    a.[TradingDay] > '2015-01-01'ORDER BY a.[TradingDay]
    """)    

# play with JYDB's constants    
cursor.execute("""
    SELECT 
    A.[SecuCode], A.[SecuAbbr], B.[EndDate], B.[T1FPE], A.[SecuCategory], A.[SecuMarket]
    FROM [JYDB].[dbo].[SecuMain] as A, [JYDB].[dbo].[RR_ProfitsForecastStatHis] as B
    WHERE A.[InnerCode] = B.[InnerCode]
    AND B.[EndDate] >= '2015-01-01'
    AND A.[SecuCategory] = 1
    AND A.[SecuMarket] in (83,90)
    AND B.[T1FPE] IS NOT NULL
    ORDER BY A.[SecuCode],B.[EndDate]
    """)
