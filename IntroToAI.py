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

#We decided it was best to merge both datasets for this piece of work and compare between them:

#Merge data and check    
training_data.drop('PassengerId', axis=1, inplace=True)
testing_data.drop('PassengerId', axis=1, inplace=True)
full_data = pd.concat([training_data, testing_data], sort=False).reset_index(drop=True)

display(full_data.columns)
display(full_data.info())



categories = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
def plotFrequency(categories):
    
    fig, axes = plt.subplots(math.ceil(len(categories)/3), 3, figsize=(29, 12))
    axes = axes.flatten()
    
    for ax, cat in zip(axes, categories):
        if cat == 'Survived':
            total = float(len(training_data[cat]))
        else:
            total = float(len(full_data[cat]))
        sns.countplot(full_data[cat], palette = 'viridis', ax=ax)
        
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 10, '{:1.2f}%' .format((height/total) * 100), ha="center")
        
            #plt.ylabel('Number of Casualties', fontsize=15, weight='bold')
        
plotFrequency(categories)



def plotsurvival(categories, data):
    
    '''A plot for bivariate analysis.'''
    
    fig, axes = plt.subplots(math.ceil(len(categories) / 3), 3, figsize=(20, 12))
    axes = axes.flatten()

    for ax, cat in zip(axes, categories):
        if cat == 'Survived':
            sns.countplot(training_data[cat], palette='viridis', ax=ax)

        else:

            sns.countplot(x=cat, data=data, hue='Survived', palette='viridis', ax=ax)
            ax.legend(title='Survived?', loc='upper right', labels=['No', 'Yes'])

        plt.ylabel('Count', fontsize=15, weight='bold')
plotsurvival(categories, training_data)


#Finding missing cabin data

fig, ax = plt.subplots(ncols=2, figsize=(20, 6))
sns.heatmap(training_data.isnull(), yticklabels=False, cbar=False, cmap='binary', ax=ax[0])
sns.heatmap(testing_data.isnull(), yticklabels=False, cbar=False, cmap='binary', ax=ax[1])

ax[0].set_title('Training Data Missing Values')
ax[1].set_title('Testing Data Missing Values')

plt.xticks(rotation=90)
plt.show()


#Feature engineering toolbox
def status(feature):
    print('Processing', feature, ': DONE')
    print(f'Shape after processing {combined.shape}')
    print('*' * 40)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    