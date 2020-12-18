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

#Modelling imports
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, cross_validate, train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, plot_roc_curve, auc
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

#import xgboost as xgb
#import lightgbm as lgb
#from mlxtend.plotting import plot_decision_regions

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
    print(f'Shape after processing {training.shape}')
    print('*' * 40)
    
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
'''
#Modelling preparation
def get_combined_data():
    # Reading train data:
    #train = pd.read_csv('train.csv')

    # Reading test data:
    #test = pd.read_csv('test.csv')

    # Extracting the targets from the training data:
    targets = train.Survived

    # Merging train data and test data for future feature engineering:
    combined = train.append(test)
    combined.reset_index(inplace=True, drop=True)

    return combined


# Filling in the missing values for Age:
def age_fill():
    global combined
    
    combined['Age'] = combined.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
    status('Age')
    return combined

def fare_fill():
    global combined

    # Filling missing values in fare:

    combined['Fare'] = combined.groupby(
        ['Pclass', 'Sex'])['Fare'].apply(lambda x: x.fillna(x.median()))
    status('fare')
    return combined

combined = get_combined_data()  # For merging train test data.
#combined = family_survival()  # For creating family survival feature.
#combined = process_family()  # For creating Family size feature.
#combined = get_titles()  # For extracting titles.
#combined = process_names()  # For one hot encoding titles.
combined = age_fill()  # For imputing missing age values.
#combined = age_binner()  # For grouping and hot encoding age ranges.
combined = fare_fill()  # For imputing fares.
# For grouping and label encoding fares, can use 'both' for keeping age with dummies or yes for just one hot encoding.
#combined = process_fare_bin(onehot='no')
# combined =scale_fare() # For scaling age values.
#combined = process_embarked()  # For imputing embarked and one hot encoding.
# combined = process_cabin() # For extracting deck info from cabins.
#combined = process_sex()  # For label encoding sex.
# combined = process_pclass() # For one hot encoding pclass.
# combined = process_ticket() # For extracting ticket info.
#combined = dropper()  # For dropping not needed features.

print(
    f'Processed everything. Missing values left: {combined.isna().sum().sum()}'
)
'''
def get_train_data():

    training = train
    training.reset_index(inplace=True, drop=True)

    return training

# Filling in the missing values for Age:
def age_fill():
    global training
    
    training['Age'] = training.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
    status('Age')
    return training

training = get_train_data()
training = age_fill()

fig, ax = plt.subplots(ncols=2, figsize=(20, 6))
sns.heatmap(training_data.isnull(), yticklabels=False, cbar=False, cmap='binary', ax=ax[0])
sns.heatmap(training.isnull(), yticklabels=False, cbar=False, cmap='binary', ax=ax[1])
 
ax[0].set_title('Training Data Missing Values')
ax[1].set_title('Testing Data Missing Values')

plt.xticks(rotation=90)
plt.show()    
    
    
    
    
    