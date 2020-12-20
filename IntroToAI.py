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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

from mlxtend.plotting import plot_decision_regions


#Import training file and check
training_data = pd.read_csv('train.csv')
index = len(training_data)
display(training_data.sample(5))

#Import test file and check
testing_data = pd.read_csv('test.csv')
display(testing_data.sample(5))



#Finding missing data (mostly cabin)

fig, ax = plt.subplots(ncols=2, figsize=(20, 6))
sns.heatmap(training_data.isnull(), yticklabels=False, cbar=False, cmap='binary', ax=ax[0])
sns.heatmap(testing_data.isnull(), yticklabels=False, cbar=False, cmap='binary', ax=ax[1])

ax[0].set_title('Training Data Missing Values')
ax[1].set_title('Testing Data Missing Values')

plt.xticks(rotation=90)
plt.show()



def status(feature):
    print('Preparing', feature, ': DONE')
    print(f'Shape after preparation {joint.shape}')
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


#We decided it was best to merge both datasets and apply our functions to both at the same time:

def get_joint_data():
    joint = train.append(test)
    joint.reset_index(inplace=True, drop=True)
    
    return joint


#Join Parents, children, siblings and spouses
def family_assembler():

    global joint
    joint['FamilySize'] = joint['Parch'] + joint['SibSp'] + 1
    
    joint['Alone'] = joint['FamilySize'].map(lambda s: 1
                                                   if s == 1 else 0)

    status('Family')
    return joint


#Calculating survival rate for members of the same family using Surname to check
def survival_of_family():
    global joint

    # Getting surnames to form families: 
    joint['Surname'] = joint['Name'].apply(lambda x: str.split(x, ",")[0])
    
    family_survival_rate = 0.5
    joint['Family_saved'] = family_survival_rate

    for grp, grp_df in joint[['Survived', 'Name', 'Surname', 'Fare', 'Ticket', 'PassengerId',
                               'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Surname', 'Fare']):

        if (len(grp_df) != 1):
            
            for ind, row in grp_df.iterrows():
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    joint.loc[joint['PassengerId'] == passID, 'Family_saved'] = 1
                elif (smin == 0.0):
                    joint.loc[joint['PassengerId'] == passID, 'Family_saved'] = 0

    for _, grp_df in joint.groupby('Ticket'):
        if (len(grp_df) != 1):
            for ind, row in grp_df.iterrows():
                if (row['Family_saved'] == 0) | (row['Family_saved'] == 0.5):
                    smax = grp_df.drop(ind)['Survived'].max()
                    smin = grp_df.drop(ind)['Survived'].min()
                    passID = row['PassengerId']
                    if (smax == 1.0):
                        joint.loc[joint['PassengerId'] == passID, 'Family_saved'] = 1
                    elif (smin == 0.0):
                        joint.loc[joint['PassengerId'] == passID, 'Family_saved'] = 0

    status('Family_saved')
    
    return joint

def separate_titles():

    title_dictionary = {
        'Capt': 'Dr/Clergy/Mil',
        'Lady': 'Honorific',
        'Dr': 'Dr/Clergy/Mil',
        'Rev': 'Dr/Clergy/Mil',
        'Jonkheer': 'Honorific',
        'Don': 'Honorific',
        'Dona': 'Honorific',
        'Sir': 'Honorific',
        'Major': 'Dr/Clergy/Mil',
        'Col': 'Dr/Clergy/Mil',
        'the Countess': 'Honorific',
        'Mme': 'Mrs',
        'Mlle': 'Miss',
        'Ms': 'Mrs',
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Miss': 'Miss',
        'Master': 'Master'
    }

    joint['Title'] = joint['Name'].map(
        lambda name: name.split(',')[1].split('.')[0].strip())

    joint['Title'] = joint.Title.map(title_dictionary)
    status('Title')
    
    return joint



def name_fix():
    
    global joint
    
    joint.drop('Name', axis=1, inplace=True)

    titles_fill = pd.get_dummies(joint['Title'], prefix='Title')
    joint = pd.concat([joint, titles_fill], axis=1)
    
    joint.drop('Title', axis=1, inplace=True)

    status('names')
    return joint



def age_fill():
    global joint
    
    joint['Age'] = joint.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
    status('Age')
    return joint


def age_range():
    global joint
    
    ranges = [0, 3, 18, 40, 70, np.inf]
    
    names = ['Under_3', '3-18', '18-40', '40-70', 'Over_70']

    joint['AgeRange'] = pd.cut(joint['Age'], ranges, labels=names)
    
    age_filled = pd.get_dummies(joint['AgeRange'], prefix='AgeRange')
    joint = pd.concat([joint, age_filled], axis=1)
    joint.drop('AgeRange', inplace=True, axis=1)
    joint.drop('Age', inplace=True, axis=1)
    status('AgeRange')
    
    return joint


def fare_fill():
    global joint

    joint['Fare'] = joint.groupby(['Pclass', 'Sex'])['Fare'].apply(lambda x: x.fillna(x.median()))
    status('fare')
    return joint


def fare_range(encode='None'):

    global joint
    
    
    ranges = [-1, 7.91, 14.454, 31, 99, 250, np.inf]
    names = [0, 1, 2, 3, 4, 5]

    joint['FareRange'] = pd.cut(joint['Fare'], ranges, labels=names).astype('int')
    if encode == 'yes':
        farerange_fill = pd.get_dummies(joint['FareRange'], prefix='FareRange')
        joint = pd.concat([joint, farerange_fill], axis=1)
        joint.drop('FareRange', inplace=True, axis=1)
        joint.drop('Fare', inplace=True, axis=1)
    elif encode == 'both':
        farerange_fill = pd.get_dummies(joint['FareRange'], prefix='FareRange')
        joint = pd.concat([joint, farerange_fill], axis=1)
        joint.drop('FareRange', inplace=True, axis=1)
    else:
        joint.drop('Fare', inplace=True, axis=1)

    status('FareRange')
    
    return joint


def embarked_fill():
    
    global joint
    
    joint.Embarked.fillna(joint.Embarked.mode()[0], inplace=True)
    
    embarked_filled = pd.get_dummies(joint['Embarked'], prefix='Embarked')
    joint = pd.concat([joint, embarked_filled], axis=1)
    joint.drop('Embarked', axis=1, inplace=True)
    status('Embarked')
    
    return joint


def gender_mapping():
    global joint
    
    joint['Sex'] = joint['Sex'].map({'male': 1, 'female': 0})
    status('Sex')
    return joint



def delete_redundant():
    joint.drop('Cabin', axis=1, inplace=True)
    joint.drop('PassengerId', inplace=True, axis=1)
    joint.drop('Surname', inplace=True, axis=1)
    joint.drop('Survived', inplace=True, axis=1)
    joint.drop('Ticket', inplace=True, axis=1)
    
    return joint



joint = get_joint_data()
joint = survival_of_family()
joint = family_assembler()
joint = separate_titles()
joint = name_fix()
joint = age_fill()
joint = age_range()
joint = fare_fill()
joint = fare_range(encode='no')
joint = embarked_fill()
joint = gender_mapping()
joint = delete_redundant()

print(f'Finished everything. Any missing values displayed here: {joint.isna().sum().sum()}')



sns.set(font_scale=1.1)
correlation_data = joint.corr()
mask = np.triu(correlation_data.corr())
plt.figure(figsize=(18, 15))
sns.heatmap(correlation_data,
            annot=True,
            fmt='.1f',
            cmap='coolwarm',
            square=True,
            mask=mask,
            linewidths=1)

plt.show()

#TEST
def find_survived():
    global joint
    y = pd.read_csv('train.csv', usecols=['Survived'])['Survived']
    X = joint.iloc[:index]
    X_test = joint.iloc[index:]

    return X, X_test, y

X, X_test, y = find_survived()

print(f'X: {X.shape}')
print(f'y: {y.shape}')
print(f'X test: {X_test.shape}')



#Modelling Section

cv = StratifiedKFold(10, shuffle=True, random_state=50)

rf = RandomForestClassifier(
    criterion='gini', n_estimators=1750, max_depth=7, min_samples_split=6, min_samples_leaf=6,
    max_features='auto', oob_score=True, random_state=50,n_jobs=-1, verbose=0)


sv = SVC(probability=True)

logreg = LogisticRegression(n_jobs=-1, solver='newton-cg')

gb = GradientBoostingClassifier(random_state=50)

estimators = [rf, gb, sv, logreg]




def check(X, y, estimators, cv):
    model_tables = pd.DataFrame()

    row_index = 0
    for est in estimators:

        MLA_name = est.__class__.__name__
        model_tables.loc[row_index, 'Model Name'] = MLA_name

        cv_results = cross_validate(
            est,
            X,
            y,
            cv=cv,
            scoring='accuracy',
            return_train_score=True,
            n_jobs=-1
        )

        model_tables.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_score'].mean()
        model_tables.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()
        model_tables.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        model_tables.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        row_index += 1

    model_tables.sort_values(by=['Test Accuracy Mean'], ascending=False, inplace=True)

    return model_tables



model_output = check(X, y, estimators, cv)
display(model_output.style.background_gradient(cmap='summer_r'))



def model_graph(models, bins):
    fig, ax = plt.subplots(figsize=(16, 8))
    g = sns.barplot('Test Accuracy Mean',
                    'Model Name',
                    data=models,
                    palette='viridis',
                    orient='h',
                    **{'xerr': models['Test Std']})
    g.set_xlabel('Test Mean Accuracy')
    g = g.set_title('Cross validation scores')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=bins))
    
model_graph(model_output, 32)



def roc_graphs(estimators, cv, X, y):

    fig, axes = plt.subplots(math.ceil(len(estimators) / 2),
                             2,
                             figsize=(50, 50))
    axes = axes.flatten()

    for ax, estimator in zip(axes, estimators):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train, test) in enumerate(cv.split(X, y)):
            estimator.fit(X.loc[train], y.loc[train])
            viz = plot_roc_curve(estimator,
                                 X.loc[test],
                                 y.loc[test],
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3,
                                 lw=1,
                                 ax=ax)
            interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1],
                linestyle='--',
                lw=2,
                color='r',
                label='Chance',
                alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr,
                mean_tpr,
                color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' %
                (mean_auc, std_auc),
                lw=2,
                alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr,
                        tprs_lower,
                        tprs_upper,
                        color='blue',
                        alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.02, 1.02],
               ylim=[-0.02, 1.02],
               title=f'{estimator.__class__.__name__} ROC')
        ax.legend(loc='lower right', prop={'size': 18})
    #plt.show()
    plt.savefig('ROC Graphs.png')
    
roc_graphs(estimators, cv, X, y)



def learning_curve_graphs(estimators, X, y, ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):

    fig, axes = plt.subplots(math.ceil(len(estimators) / 2), 2, figsize=(50, 50))
    
    axes = axes.flatten()

    for ax, estimator in zip(axes, estimators):

        ax.set_title(f'{estimator.__class__.__name__} Learning Curve')
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Score')

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)


        ax.fill_between(train_sizes,
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std,
                        alpha=0.1,
                        color='r')
        ax.fill_between(train_sizes,
                        test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std,
                        alpha=0.1,
                        color='g')
        ax.plot(train_sizes,
                train_scores_mean,
                'o-',
                color='red',
                label='Training score')
        ax.plot(train_sizes,
                test_scores_mean,
                'o-',
                color='green',
                label='Cross-validation score')
        ax.legend(loc='best')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=24))

    #plt.show()
    plt.savefig('Learning curve graphs.png')


learning_curve_graphs(estimators, X, y, ylim=None, cv=cv, n_jobs=-1,
                    train_sizes=np.linspace(.1, 1.0, 10))