# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:19:41 2016

@author: lixiaolong
"""

import pyodbc
import pandas as pd
import numpy as np
import datetime

def main():
    """
    basic DB
    translated from SAS code
    """
    startdatestr = 'yyyy-mm-dd'
    enddatestr = 'yyyy-mm-dd'
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
