# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:16:46 2016

@author: lixiaolong
"""

from reportlab.pdfgen import canvas

def hello(c):
    c.drawString(100,100,"Hello World")
    
c = canvas.Canvas("hello.pdf")
hello(c)
c.showPage()
c.save()