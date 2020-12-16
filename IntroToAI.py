# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:15:17 2020

@author: JohnA
"""

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scipy import interp
import math
from scipy.stats import norm
from scipy import stats

#Don't know if necessary
import warnings
warnings.filterwarnings('ignore') # Disabling warnings for clearer outputs.
pd.options.display.max_columns = 50 # Pandas option to increase max number of columns to display.
plt.style.use('ggplot') # Setting default plot style.

#Import training file and check
training_data = pd.read_csv('train.csv')
index = len(training_data)
display(training_data.sample(5))

#Import test file and check
testing_data = pd.read_csv('test.csv')
index = len(testing_data)
display(testing_data.sample(5))