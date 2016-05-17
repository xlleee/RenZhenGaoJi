# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:29:57 2016

@author: lixiaolong
"""

def findnum(x):
    i = 1
    while i <= x:
        if fnum(i) == 45:
            print(i)
        i += 1
    
    
    
    
def fnum(n):
    count = 0
    i = 1
    while i <= n:
        if int(n/i) == (n/i):
            count += 1
        i += 1
    return count
    
def findprime():
    prime = [2,3,5,7,11,13,17,23,29,31,37,41,43,47,53]
    for i in prime:
        for j in prime:
            for q in prime:
                if i+j+q == 53 and i!=j and j!=q and i!=q:
                    print(str(i) + ',' + str(j) + ',' + str(q))