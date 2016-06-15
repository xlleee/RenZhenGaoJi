# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:28:07 2016

@author: lixiaolong
"""

import numpy as np
import matplotlib.pyplot as plt

# stack
def fnx():
    return np.random.randint(5, 50, 10)

y = np.row_stack((fnx(), fnx(), fnx()))
x = np.arange(10)

y1, y2, y3 = fnx(), fnx(), fnx()

fig, ax = plt.subplots()
ax.stackplot(x, y)
plt.show()

fig, ax = plt.subplots()
ax.stackplot(x, y1, y2, y3)
plt.show()

fig, ax = plt.subplots()
ax.plot(x, y1)
ax.stackplot(x, y2, y3)
plt.show()

# two y axis
x = np.arange(0., np.e, 0.01)
y1 = np.exp(-x)
y2 = np.log(x)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1,'r',label="right");
ax1.legend(loc=1)
ax1.set_ylabel('Y values for exp(-x)');
ax2 = ax1.twinx() # this is the important function
ax2.plot(x, y2, 'g',label = "left")
ax2.legend(loc=2)
ax2.set_xlim([0, np.e]);
ax2.set_ylabel('Y values for ln(x)');
ax2.set_xlabel('Same X for both exp(-x) and ln(x)');
plt.show()

# hist
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x1 = mu + sigma * np.random.randn(10000)
x2 = mu + 50 + sigma * np.random.randn(10000)
num_bins = 50
n1, bins1, patches1 = plt.hist(x1, num_bins, 
                               normed=1, 
                               facecolor='green', alpha=0.3,
                               histtype = 'stepfilled')
n1, bins1, patches1 = plt.hist(x2, num_bins, 
                               normed=1, 
                               facecolor='red', alpha=0.3,
                               histtype = 'stepfilled')
plt.table(cellText = [['a','b','c'],[1, 2, 3]],
          rowLabels = ['1 row', '2 row'],
            colLabels = ['1 col', '2 col', '3 col'],
            loc = 'bottom',
            bbox = [0, -0.25, 1, 0.15])

                               
# adding text, legends, table ....





