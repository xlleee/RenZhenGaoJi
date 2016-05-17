# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:12:13 2016

@author: lixiaolong
"""

from multiprocessing import Pool

def f(x):
    print(x*x)
    return x*x

def main(tt):
    with Pool(5) as p:
        print(p.map(f, range(1,tt)))
        
        
if __name__ == '__main__':
    main()
