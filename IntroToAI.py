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

import xgboost as xgb
import lightgbm as lgb
from mlxtend.plotting import plot_decision_regions

#Don't know if necessary
import warnings
warnings.filterwarnings('ignore') # Disabling warnings for clearer outputs.
pd.options.display.max_columns = 50 # Pandas option to increase max number of columns to display.
plt.style.use('ggplot') # Setting default plot style.

#Import training file and check
training_data = pd.read_csv('train.csv')
index = len(training_data)
#display(training_data.sample(5))

#Import test file and check
testing_data = pd.read_csv('test.csv')
index = len(testing_data)
#display(testing_data.sample(5))

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


def get_train_data():

    training = train
    training.reset_index(inplace=True, drop=True)

    return training

def get_test_data():

    testing = test
    testing.reset_index(inplace=True, drop=True)

    return testing


# Filling in the missing values for Age:
def age_fill():
    global training
    
    training['Age'] = training.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
    status('Age')
    return training

# Filling in the missing values for Age:
def age_fill2():
    global testing
    
    testing['Age'] = testing.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
    status('Age')
    return testing


# Ranging and grouping Ages
def age_range():
    global training
    
    bins = [0, 3, 18, 40, 70, np.inf]
    
    names = ['Under_3', '3-18', '18-40', '40-70', 'Over_70']

    training['Age Range'] = pd.cut(training['Age'], bins, labels=names)
    
    age_dummies = pd.get_dummies(training['Age Range'], prefix='Age Range')
    training = pd.concat([training, age_dummies], axis=1)
    training.drop('Age Range', inplace=True, axis=1)
    training.drop('Age', inplace=True, axis=1)
    status('Age Bins')
    
    return training

def age_range2():
    global testing
    
    bins = [0, 3, 18, 40, 70, np.inf]
    
    names = ['Under_3', '3-18', '18-40', '40-70', 'Over_70']

    testing['Age Range'] = pd.cut(testing['Age'], bins, labels=names)
    
    age_dummies = pd.get_dummies(testing['Age Range'], prefix='Age Range')
    testing = pd.concat([testing, age_dummies], axis=1)
    testing.drop('Age Range', inplace=True, axis=1)
    testing.drop('Age', inplace=True, axis=1)
    status('Age Bins')
    
    return testing


# Filling missing values in fare:
def fare_fill():
    global testing

    testing['Fare'] = testing.groupby(['Pclass', 'Sex'])['Fare'].apply(lambda x: x.fillna(x.median()))
    status('fare')
    return testing


# Filling missing embarked values with the most frequent one:
def embarked_fill():
    
    global training
    
    training.Embarked.fillna(training.Embarked.mode()[0], inplace=True)
    
    # One hot encoding.
    
    embarked_dummies = pd.get_dummies(training['Embarked'], prefix='Embarked')
    training = pd.concat([training, embarked_dummies], axis=1)
    training.drop('Embarked', axis=1, inplace=True)
    status('Embarked')
    
    return training


def gender_mapping():
    global training
    
    # Mapping string values with numerical ones.
    
    training['Sex'] = training['Sex'].map({'male': 0, 'female': 1})
    status('Sex')
    return training


def gender_mapping2():
    global testing
    
    # Mapping string values with numerical ones.
    
    testing['Sex'] = testing['Sex'].map({'male': 0, 'female': 1})
    status('Sex')
    return testing


del train['Cabin']
del test['Cabin']

training = get_train_data()
testing = get_test_data()
training = age_fill()
testing = age_fill2()
training = age_range()
testing = age_range2()
testing = fare_fill()
training = embarked_fill()
training = gender_mapping()
testing = gender_mapping2()


fig, ax = plt.subplots(ncols=2, figsize=(30, 15))
#fig.tight_layout()
sns.heatmap(training_data.isnull(), yticklabels=False, cbar=False, cmap='binary', ax=ax[0])
sns.heatmap(training.isnull(), yticklabels=False, cbar=False, cmap='binary', ax=ax[1])
 
ax[0].set_title('Training Data with missing Values')
ax[1].set_title('Training Data after processing age')

plt.xticks(rotation=90)
plt.show()    
    
fig, ax = plt.subplots(ncols=2, figsize=(30, 15))
#fig.tight_layout()
sns.heatmap(testing_data.isnull(), yticklabels=False, cbar=False, cmap='binary', ax=ax[0])
sns.heatmap(testing.isnull(), yticklabels=False, cbar=False, cmap='binary', ax=ax[1])
 
ax[0].set_title('Testing Data with missing Values')
ax[1].set_title('Testing Data after processing age')

plt.xticks(rotation=90)
plt.show()

#display(training.sample(10))
training_copy = training.copy()
del training_copy['PassengerId']

sns.set(font_scale=1.1)
correlation_train = training_copy.corr()
mask = np.triu(correlation_train.corr())
plt.figure(figsize=(18, 15))
sns.heatmap(correlation_train,
            annot=True,
            fmt='.1f',
            cmap='coolwarm',
            square=True,
            mask=mask,
            linewidths=1)

plt.show()
#TEST
def recover_train_test_target():
    global training
    y = pd.read_csv('train.csv', usecols=['Survived'])['Survived']
    X = training.iloc[:index]
    X_test = training.iloc[index:]

    return X, X_test, y

X, X_test, y = recover_train_test_target()

print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
print(f'X_test shape: {X_test.shape}')


#Modelling Section
cv = StratifiedKFold(10, shuffle=True, random_state=42)

rf = RandomForestClassifier(criterion='gini',
                            n_estimators=1750,
                            max_depth=7,
                            min_samples_split=6,
                            min_samples_leaf=6,
                            max_features='auto',
                            oob_score=True,
                            random_state=42,
                            n_jobs=-1,
                            verbose=0)

lg = lgb.LGBMClassifier(max_bin=4,
                        num_iterations=550,
                        learning_rate=0.0114,
                        max_depth=3,
                        num_leaves=7,
                        colsample_bytree=0.35,
                        random_state=42,
                        n_jobs=-1)

xg = xgb.XGBClassifier(
    n_estimators=2800,
    min_child_weight=0.1,
    learning_rate=0.002,
    max_depth=2,
    subsample=0.47,
    colsample_bytree=0.35,
    gamma=0.4,
    reg_lambda=0.4,
    random_state=42,
    n_jobs=-1,
)

sv = SVC(probability=True)

logreg = LogisticRegression(n_jobs=-1, solver='newton-cg')

gb = GradientBoostingClassifier(random_state=42)

gnb = GaussianNB()

mlp = MLPClassifier(random_state=42)

estimators = [rf, lg, xg, gb, sv, logreg, gnb, mlp]


def model_check(X, y, estimators, cv):
    model_table = pd.DataFrame()

    row_index = 0
    for est in estimators:

        MLA_name = est.__class__.__name__
        model_table.loc[row_index, 'Model Name'] = MLA_name
        #    model_table.loc[row_index, 'MLA Parameters'] = str(est.get_params())

        cv_results = cross_validate(
            est,
            X,
            y,
            cv=cv,
            scoring='accuracy',
            return_train_score=True,
            n_jobs=-1
        )

        model_table.loc[row_index, 'Train Accuracy Mean'] = cv_results[
            'train_score'].mean()
        model_table.loc[row_index, 'Test Accuracy Mean'] = cv_results[
            'test_score'].mean()
        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        row_index += 1

    model_table.sort_values(by=['Test Accuracy Mean'],
                            ascending=False,
                            inplace=True)

    return model_table

raw_models = model_check(X, y, estimators, cv)
display(raw_models.style.background_gradient(cmap='summer_r'))

