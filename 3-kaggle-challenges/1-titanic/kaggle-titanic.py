# -*- coding: utf-8 -*-
"""
Analysis of Titanic dataset for the Getting Started Kaggle Competition: predict
the survival of some passengers based on some relevant features. After data
preprocessing and analysis, the following methods are employed:
    
(1) Logistic regression

(2) KNN

(3) SVM

(4) Random forest

(5) Logistic regression: in this case, 2 different models are trained, the first 
                         on the rows with available age, the second on the rows 
                         with no available age

@author: Andrea Boselli
"""

#%% Relevant libraries
import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
from datetime                import datetime
from sklearn.ensemble        import RandomForestClassifier
from sklearn.experimental    import enable_iterative_imputer
from sklearn.impute          import IterativeImputer
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import confusion_matrix
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing   import MinMaxScaler
from sklearn.svm             import SVC


#%% Main settings
Data_filepath  = os.getcwd()    # path to the current dataset
Train_filename = 'train.csv'    # filename of the training data
Test_filename  = 'test.csv'     # filename of the test     data
method_name    = 'RandomForest' # name of the employed method
plots_heights  = 3              # heights of subplots
Verbose        = False          # verbose mode
Debug          = False          # debug   mode


#%% Settings for each employed method

# Logistic regression model without age as a feature
if method_name == 'LogisticReg': 
    age_as_a_feature = False                                                   # set true if the age is considered in the model
    model = LogisticRegression                                                 # function of the employed classification model 
    selected_cols = ['Sex', 'SibSp', 'Fare', 'Pclass']                         # list with the names of the selected features
    search_grid = {
                    "C":            np.logspace(start=np.log10(0.001),stop=np.log10(10),num=50,endpoint=True),
                    "penalty":      ['l1','l2'], 
                    "solver":       ['liblinear'], 
                    "class_weight": [None, 'balanced']
                  }                                                            # dictionary with all the settings considered in the grid search


# K-nearest neighbor method
if method_name == 'KNN':
    age_as_a_feature = False
    model = KNeighborsClassifier
    selected_cols = ['Sex','Fare', 'Emb_Q', 'Pclass']
    search_grid = {
                    "n_neighbors": np.arange(1,11),
                    "weights":     ['uniform','distance'],
                    "p":           [1,2]
                  }


# Support vector machine with different kernel functions
if method_name == 'SVM':
    age_as_a_feature = False
    model = SVC
    selected_cols = ['Sex','Fare', 'Pclass']
    search_grid = {
                    "C":            np.power(10, np.arange(start=-3,stop=2,dtype='float16')),
                    "degree":       [1,2,3],
                    "gamma":        ['scale','auto'],
                    "kernel":       ['linear','poly','rbf','sigmoid'],
                    "class_weight": [None, 'balanced']
                  }
    

# Random forest classifier
if method_name == 'RandomForest':
   age_as_a_feature = False
   model = RandomForestClassifier
   selected_cols = ['Sex', 'Fare', 'Pclass']
   search_grid = {
                    "n_estimators": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100],
                    "criterion":    ['gini','entropy'],
                    "bootstrap":    [True, False],
                    "random_state": [1],
                    "class_weight": [None, 'balanced', 'balanced_subsample']
                 }


# Logistic regression model where 2 models are trained, depending on the age availability
if method_name == 'LogisticReg_WithAge':
    age_as_a_feature = True
    search_grid = {
                    "C":            np.logspace(start=np.log10(0.001),stop=np.log10(10),num=50,endpoint=True),
                    "penalty":      ['l1','l2'], 
                    "solver":       ['liblinear'], 
                    "class_weight": [None, 'balanced']
                  }
    

#%% User-defined functions

def feature_selection(data, data_test, selected_cols):
    """
    Hold only the selected columns in both training and test data, and convert
    all tables into NumPy arrays for subsequent analysis
    """
    X      = data     [selected_cols].to_numpy()
    y      = data     ['Survived'   ].to_numpy()
    X_test = data_test[selected_cols].to_numpy()
    return X,y,X_test


def model_assessment(model_grid, X, y, plots_heights):
    """
    Compute the prediction of the best model after grid search, show method
    accuracy and plot the confusion matrix with respect to the training data
    """
    
    # Prediction computation
    y_pred = model_grid.best_estimator_.predict(X)
    
    # Chosen settings and accuracy
    print("Accuracy for the selected model: " + str(model_grid.best_score_))
    print("Settings for the selected model:\n",     model_grid.best_params_)
    
    # Confusion matrix plot
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(1, 1, figsize = (plots_heights,plots_heights), tight_layout=True, dpi = 500)
    sns.heatmap(cm/np.sum(cm), annot=True, cmap='plasma', ax = ax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title ('Confusion matrix')
    ax.set_xlabel('predicted class')
    ax.set_ylabel('real class')


def perform_prediction(model_grid, X_test, data_test_ID, data_filepath, debug):
    """
    Perform prediction on the test data save the results (id - prediction) in a
    suitably named .csv file
    """
    
    # Generate prediction on test data
    y_test = model_grid.best_estimator_.predict(X_test)
    
    # table and filename 
    prediction_table = pd.DataFrame({"PassengerId": data_test_ID, "Survived": y_test})
    prediction_filename = 'predictions_Boselli_' + model_grid.estimator.__str__().replace('()','') + '_' + f'{datetime.now():%Y-%m-%d_%H-%M-%S%z}' + '.csv'

    # Save prediction
    if(not debug):
        prediction_table.to_csv(path_or_buf=os.path.join(data_filepath,prediction_filename),index=False)


#%% Load, analyse and process training data
data      = pd.read_csv(filepath_or_buffer = os.path.join(Data_filepath,Train_filename))
data_test = pd.read_csv(filepath_or_buffer = os.path.join(Data_filepath,Test_filename ))

# Check the fraction of non-nan entries for each column
if(Verbose):
    nan_fraction       = data.notna().sum()/len(data)
    print('The fraction of non-nan entries for each column of the training data is:')
    print(nan_fraction)
    nan_fraction_test  = data_test.notna().sum()/len(data_test)
    print('The fraction of non-nan entries for each column of the test data is:')
    print(nan_fraction_test)

# Discard the uninformative columns and the cabin column, that has very few informative entries
data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
# Discard the (few) record with NaNs for embarked 
data = data.dropna(subset=['Embarked'],axis=0)

# Print features proportions barplots for some columns, and histograms for others
if(Verbose):
    cols_proportions = ['Survived','Pclass','Sex','SibSp','Parch','Embarked']
    cols_histograms  = ['Age','Fare']
    
    fig, axs = plt.subplots(1, len(cols_histograms), figsize = (plots_heights*len(cols_histograms),plots_heights), tight_layout=True, dpi = 500)
    fig.suptitle('Histograms')
    for n in range(len(cols_histograms)):
        axs[n].hist(data[[cols_histograms[n]]].dropna(), density = True, bins = 20)
        axs[n].set_title ('Histogram - ' + cols_histograms[n])
        axs[n].set_xlabel(cols_histograms[n])
        axs[n].set_ylabel('Density')
        axs[n].grid(visible=True)
        
    fig, axs = plt.subplots(1, len(cols_proportions), figsize = (plots_heights*len(cols_proportions),plots_heights), tight_layout=True, dpi = 500)
    fig.suptitle('Barplots')
    for n in range(len(cols_proportions)):
        col_items = data[cols_proportions[n]].value_counts().to_frame().reset_index().rename(columns={'index':'value', cols_proportions[n]:'count'})
        axs[n].bar(x=col_items['value'], height=col_items['count']/len(data), tick_label = col_items['value'])
        axs[n].set_title(cols_proportions[n])
        axs[n].set_ylabel('Relative frequency')
        axs[n].grid(visible=True)
    
# Transform categorical data
data['Sex']   = (data['Sex'     ] == 'male')*1 # 1 if male, 0 if female
data['Emb_C'] = (data['Embarked'] == 'C'   )*1 # 1 if embarked from C
data['Emb_Q'] = (data['Embarked'] == 'Q'   )*1 # 1 if embarked from Q
data = data.drop(['Embarked'],axis=1)
    
# Print the pairplot of the data and the correlation matrix
if(Verbose):
    pp = sns.pairplot(data=data,hue='Survived',kind="hist",diag_kind='hist',corner=True) # pairplot
    pp.fig.suptitle('Pairplot',fontsize=32)
    pp.fig.set_dpi(500)
    fig,ax = plt.subplots(1, 1, tight_layout=True, dpi = 500)    # correlation matrix
    sns.heatmap(data.corr(), linewidth=0.5, cmap='plasma',ax=ax) 
    ax.set_title('Correlation matrix')
    ax.set_aspect('equal', adjustable='box')
    
# Standardize quantitative columns
scaler = MinMaxScaler()
data[['Age','Fare','SibSp','Parch']] = scaler.fit_transform(data[['Age','Fare','SibSp','Parch']])
data['Pclass'] = (4-data['Pclass'])/3


#%% Preprocess test data
data_test_ID=data_test['PassengerId']
data_test = data_test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
data_test['Sex']   = (data_test['Sex'     ] == 'male')*1
data_test['Emb_C'] = (data_test['Embarked'] == 'C'   )*1
data_test['Emb_Q'] = (data_test['Embarked'] == 'Q'   )*1
data_test = data_test.drop(['Embarked'],axis=1)
data_test[['Age','Fare','SibSp','Parch']] = scaler.transform(data_test[['Age','Fare','SibSp','Parch']])
data_test['Pclass'] = (4-data_test['Pclass'])/3

# Impute missing values for the fare column
imp = IterativeImputer(max_iter=10, random_state=0).fit(data_test)
fare_idx = np.where(data_test.columns == 'Fare')[0][0]
data_test['Fare'] = imp.transform(data_test)[:,fare_idx]


#%% Application of a method that doesn't employ the age as a feature

if(not age_as_a_feature):
    X,y,X_test = feature_selection(data, data_test, selected_cols)             # features selection
    model_grid = GridSearchCV(model(), param_grid = search_grid)               # grid search instantiation
    model_grid.fit(X,y)                                                        # model training with grid search
    model_assessment(model_grid, X, y, plots_heights)                          # model assessment
    perform_prediction(model_grid, X_test, data_test_ID, Data_filepath, Debug) # prediction generation


#%% LogisticReg1: logistic regression model with separation between records with available age and record with unavailable age

if method_name == 'LogisticReg_WithAge':
    
    # Data splitting based on availability of age
    data_age_y = data[~data['Age'].isnull()]                      # data with    age
    data_age_n = data[ data['Age'].isnull()].drop(['Age'],axis=1) # data without age
    X_age_y = data_age_y.drop(['Survived'],axis=1).to_numpy()
    X_age_n = data_age_n.drop(['Survived'],axis=1).to_numpy()
    y_age_y = data_age_y      ['Survived'].to_numpy()
    y_age_n = data_age_n      ['Survived'].to_numpy()
    
    # Model training with grid search
    logistic_regression_y=GridSearchCV(LogisticRegression(),param_grid = search_grid)
    logistic_regression_n=GridSearchCV(LogisticRegression(),param_grid = search_grid)
    logistic_regression_y.fit(X_age_y,y_age_y)
    logistic_regression_n.fit(X_age_n,y_age_n)
    
    # Model assessment: accuracy, settings, confusion matrix
    y_age_y_pred = logistic_regression_y.best_estimator_.predict(X_age_y)
    y_age_n_pred = logistic_regression_n.best_estimator_.predict(X_age_n)
    print("Accuracy for the selected model, with    available age: " + str(logistic_regression_y.best_score_))
    print("Accuracy for the selected model, without available age: " + str(logistic_regression_n.best_score_))
    print("Settings for the selected model, with    available age:\n", logistic_regression_y.best_params_)
    print("Settings for the selected model, without available age:\n", logistic_regression_n.best_params_)
    cm_age_y = confusion_matrix(y_age_y, y_age_y_pred)
    cm_age_n = confusion_matrix(y_age_n, y_age_n_pred)
    cm_age   = confusion_matrix(np.append(y_age_y,y_age_n), np.append(y_age_y_pred,y_age_n_pred))
    
    fig, axs = plt.subplots(1, 3, figsize = (plots_heights*3,plots_heights), tight_layout=True, dpi = 500)
    fig.suptitle('Confusion matrices')
    sns.heatmap(cm_age_y/np.sum(cm_age_y), annot=True, cmap='plasma', ax = axs[0])
    sns.heatmap(cm_age_n/np.sum(cm_age_n), annot=True, cmap='plasma', ax = axs[1])
    sns.heatmap(cm_age  /np.sum(cm_age)  , annot=True, cmap='plasma', ax = axs[2])
    axs[0].set_title ('Age available')
    axs[1].set_title ('Age unavailable')
    axs[2].set_title ('Training data')
    for ax in axs:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('predicted class')
        ax.set_ylabel('real class')
        
    # Perform prediction
    X_test_age_y = data_test[~data_test['Age'].isnull()].to_numpy()
    X_test_age_n = data_test[ data_test['Age'].isnull()].drop(['Age'],axis=1).to_numpy()
    y_test_age_y = logistic_regression_y.best_estimator_.predict(X_test_age_y)
    y_test_age_n = logistic_regression_n.best_estimator_.predict(X_test_age_n)
    id_test_age_y = data_test_ID[~data_test['Age'].isnull()]
    id_test_age_n = data_test_ID[ data_test['Age'].isnull()]
    prediction_table = pd.DataFrame({"PassengerId": np.append(id_test_age_y,id_test_age_n),
                                     "Survived":    np.append( y_test_age_y, y_test_age_n) })
    prediction_filename = 'predictions_Boselli_' + method_name + '_' + f'{datetime.now():%Y-%m-%d_%H-%M-%S%z}' + '.csv'
    if(not Debug):
        prediction_table.to_csv(path_or_buf=os.path.join(Data_filepath,prediction_filename),index=False)