# -*- coding: utf-8 -*-
"""
Analysis of the Spaceship Titanic dataset for the Getting Started Kaggle Competition: 
predict the survival of some passengers based on some relevant features. As an important 
part of the records lacked of some values, great importance has been given to the dataset 
analysis (contingency tables, countplots, histograms, etc...), that allowed to impute
many of the missing values

@author: Andrea Boselli
"""

#%% Relevant libraries
import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import tensorflow        as tf
from datetime                import datetime
from sklearn.preprocessing   import MinMaxScaler
from scipy.stats.contingency import crosstab
from tensorflow              import keras
from tensorflow.keras        import layers


#%% Main settings
Data_filepath  = os.getcwd() # path to the current dataset
Train_filename = 'train.csv' # filename of the training data
Test_filename  = 'test.csv'  # filename of the test     data
plots_heights  = 3           # heights of subplots
method_number  = 1           # name of the employed method
Verbose1       = False       # verbose mode for images
Verbose2       = False       # verbose mode for imputing phase
Debug          = False       # debug mode


#%% Settings for each employed method

if(method_number == 1): # ANN approach
    m1_settings = {}
    m1_settings['optimizer']  = 'adam'
    m1_settings['loss']       = 'binary_crossentropy'
    m1_settings['batch_size'] = 16
    m1_settings['n_epochs']   =  5

    
#%% User-defined functions

def find_n_nans(data):
    """For each column of the data DataFrame, retrieve the total number of nans"""
    return data.isnull().sum()

def convert_into_onehot(data, col_name, col_labels):
    """
    Given the data table, the column name and its labels, create a dummy column for each label 
    (except the last one) in the data table
    """
    for label in col_labels[0:-1]:
        label_col_name = col_name+label.replace(' ','')[0:7].title()          # new column name
        label_col      = (data[col_name]==label).replace({True: 1, False: 0}) # new column
        label_col[data[col_name].isna()] = np.nan                             # insertion of nans
        data.insert(loc=int(np.where(data.columns==col_name)[0][0]), column=label_col_name, value=label_col)
        
def count_distributions_per_group(data,input_col, cols_groups):
    """
    In the data DataFrame, for each different value of input_col column, check
    the number of different records at different groups of columns; the list of 
    columns that are check simultaneously are stored in the cols_groups list
    """
    for cols_group in cols_groups:
        nlabels_per_group = data[[input_col]+cols_group].dropna().drop_duplicates().groupby(input_col).size()
        nlabels_distr = nlabels_per_group.value_counts()
        print("\nCurrent variables: ",cols_group,"\nThe distribution of different values combinations per "+input_col+" is:\n",nlabels_distr,'\n',sep='')

def impute_category_based_on_group(data,group_col,imputed_cols):
    """
    In data DataFrame, impute the values in columns imputed_cols based on the
    column group_col: if the values in imputed_cols are missing, they are searched
    among the records with same group_col; in case of multiple possible values
    or absence of such records, no value is imputed.
    """
    lookup   = data[[group_col]+imputed_cols].dropna().drop_duplicates().drop_duplicates(subset=group_col,keep=False)
    nan_idxs = data[data[imputed_cols[0]].isna()][[group_col]]
    imp_values = nan_idxs.set_index(group_col).join(other=lookup.set_index(group_col)).set_index(nan_idxs.index)
    data.loc[data[imputed_cols[0]].isna(),imputed_cols] = imp_values
        

#%% Load, analyse and process training data
data      = pd.read_csv(filepath_or_buffer = os.path.join(Data_filepath,Train_filename))
data_test = pd.read_csv(filepath_or_buffer = os.path.join(Data_filepath,Test_filename ))
n_data      = len(data)      # number of training samples
n_data_test = len(data_test) # number of test     samples

# Check the fraction of non-nan entries for each column
nan_fraction      = data     .notna().sum()/n_data
nan_fraction_test = data_test.notna().sum()/n_data_test
if(Verbose1):
    print('\nThe fraction of non-nan entries for each column of the training data is:')
    print(nan_fraction)
    print('\nThe fraction of non-nan entries for each column of the test data is:')
    print(nan_fraction_test)
    
# Extract passenger group from id, extract cabin deck and side from cabin, extract first name and surname from
data.insert(loc=int(np.where(data.columns=='PassengerId')[0][0]), column='PassengerGroup', value=data['PassengerId'].str.split('_').str[0])
data.insert(loc=int(np.where(data.columns=='Cabin'      )[0][0]), column='CabinDeck',      value=data['Cabin'      ].str.split('/').str[0])
data.insert(loc=int(np.where(data.columns=='Cabin'      )[0][0]), column='CabinSide',      value=data['Cabin'      ].str.split('/').str[2])
data.insert(loc=int(np.where(data.columns=='Name'       )[0][0]), column='FirstName',      value=data['Name'       ].str.split(' ').str[0])
data.insert(loc=int(np.where(data.columns=='Name'       )[0][0]), column='LastName',       value=data['Name'       ].str.split(' ').str[1])
data.drop(columns=['Name','PassengerId','Cabin'],inplace=True)

# List the columns that are employed in the different operations / plots
columns_categorical        = ['HomePlanet','CryoSleep','CabinDeck','CabinSide','Destination','VIP','Transported']
columns_categorical_input  = [column for column in columns_categorical if column != 'Transported']
columns_categorical_bool   = ['CryoSleep','VIP']
columns_categorical_nobool = [column for column in columns_categorical_input if column not in columns_categorical_bool]
columns_continuous         = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

# Extract the labels of each categorical variable
categories_labels = {}
for column in columns_categorical:
    categories_labels[column] = data[column].value_counts().to_frame().reset_index().rename(columns={'index':'value', column:'count'})
    if(Verbose1):
        print('\nThe labels of '+column+' column are:\n', categories_labels[column], sep='')
    
# Plot the proportion barplots for the categorical variables and the histograms for continuous variables
if(Verbose1):
    # Barplots of categorical variables
    fig,axs = plt.subplots(1,len(columns_categorical), figsize=(plots_heights*len(columns_categorical),plots_heights), tight_layout=True, dpi=800)
    fig.suptitle('Relative frequences of categorical variables')
    for n in range(len(columns_categorical)):
        curr_col = columns_categorical[n]
        curr_labels = categories_labels[curr_col]
        axs[n].bar(x=curr_labels['value'], height=curr_labels['count']/n_data, tick_label=curr_labels['value'], color='deepskyblue')
        if(curr_col == 'Destination'):
            axs[n].tick_params(labelsize='x-small')
        axs[n].set_title(curr_col)
        axs[n].set_ylabel('Relative frequency')
        axs[n].grid(visible=True)
        axs[n].set_axisbelow(True)
        
    # Barplots of categorical variables, grouped with respect to target category
    fig,axs = plt.subplots(1,len(columns_categorical_input), figsize=(plots_heights*len(columns_categorical_input), plots_heights), 
                           tight_layout=True, dpi=800)
    fig.suptitle('Relative frequences of categorical variables for each target value')
    for n in range(len(columns_categorical_input)):
        curr_col = columns_categorical_input[n]
        curr_frequencies = data.groupby([curr_col, 'Transported']).size().reset_index().rename(columns={0:'count'})
        curr_frequencies['count'] = curr_frequencies['count'] / np.sum(curr_frequencies['count'])
        sns.barplot(data=curr_frequencies, x=curr_col, y='count', hue='Transported', ax=axs[n])
        if(curr_col == 'Destination'):
            axs[n].tick_params(labelsize='x-small')
        axs[n].legend(fontsize='x-small')
        axs[n].set_ylabel('Relative frequency')
        axs[n].grid(visible=True)
        axs[n].set_axisbelow(True)
        
    # Barplots of the proportions of zeros in all the continuous variables
    fig, axs = plt.subplots(1, len(columns_continuous), figsize=(plots_heights*len(columns_continuous),plots_heights), tight_layout=True, dpi=800)
    fig.suptitle('Proportions of 0 in the continuous variables')
    for n in range(len(columns_continuous)):
        curr_col = columns_continuous[n]
        axs[n].bar(x=[1,2], height= [sum(data[curr_col]==0)/n_data, sum(data[curr_col]>0)/n_data], tick_label=['= 0','> 0'], color='deepskyblue')
        axs[n].set_title (curr_col)
        axs[n].set_ylabel('Relative frequency')
        axs[n].grid(visible=True)
        axs[n].set_axisbelow(True)
    
    # Histograms of the continuous variables (after log transformations and excluding zero values)
    fig, axs = plt.subplots(1, len(columns_continuous), figsize=(plots_heights*len(columns_continuous),plots_heights), tight_layout=True, dpi=800)
    fig.suptitle('Histograms of continuous variables (after log transformations and excluding zero values)')
    for n in range(len(columns_continuous)):
        curr_col = columns_continuous[n]
        axs[n].hist(np.log10(data[data[curr_col] > 0][curr_col]+1), density=True, color='deepskyblue', bins=60)
        axs[n].set_title ('Histogram - ' + curr_col)
        axs[n].set_ylabel('Density')
        axs[n].set_xlabel(curr_col)
        axs[n].grid(visible=True)
        axs[n].set_axisbelow(True)
        
# Plot the contingency matrices of all the couples of categorical data
if(Verbose2):
    fig, axs = plt.subplots(len(columns_categorical)-1, len(columns_categorical)-1,
                            figsize=(plots_heights*(len(columns_categorical)-1),plots_heights*(len(columns_categorical)-1)), tight_layout=True, dpi=800)
    fig.suptitle('Contingency matrices of the categorical data',fontsize=32)
    for i in range(len(columns_categorical)-1):
        for j in range(len(columns_categorical)-1):
            if(j <= i):
                # Create the contingency matrix
                col_i = columns_categorical[i+1]
                col_j = columns_categorical[j]
                data_no_nan = data[[col_i,col_j]].dropna()
                cont_table = crosstab(data_no_nan[col_i], data_no_nan[col_j])
                
                # Plot the contingency matrix
                sns.heatmap(data = cont_table[1]/len(data_no_nan), cmap='plasma', ax = axs[i,j],
                            yticklabels = [str(lab_name)[0:3] for lab_name in cont_table[0][0]],
                            xticklabels = [str(lab_name)[0:3] for lab_name in cont_table[0][1]],
                            annot = True, cbar = False, annot_kws={'fontsize':'xx-small', 'rotation':45})
                axs[i,j].set_ylabel(col_i)
                axs[i,j].set_xlabel(col_j)
                axs[i,j].set_aspect('equal', adjustable='box')
            else:
                axs[i,j].axis('off')
    
# For each continuous variable, sum 1 and convert to logarithmic scale
for column in columns_continuous:
    data[column] = np.log10(data[column]+1)
    
# Plot the pairplot of the continuous variables and the correlation matrix of all the variables
if(Verbose1):
    for column in columns_categorical:
        pp = sns.pairplot(data=data, vars=columns_continuous, hue=column,kind="hist",diag_kind='hist',corner=True)
        pp.fig.suptitle("Pairplot of continuous variables, for each '" + column + "' group",fontsize=32)
        pp.fig.set_dpi(800)
    
# Transform True/False categorical variables into 1/0 values
for column in columns_categorical_bool:
    data[column] = data[column].replace({True: 1, False: 0})
    
# Transform the other categorical variables into one-hot-encoded variables
for column in columns_categorical_nobool:
    convert_into_onehot(data, column, categories_labels[column]['value'])
    data.drop(columns=column,inplace=True)

# Scale continuous variables into [0,1] interval
cont_scaler = MinMaxScaler()
data[columns_continuous] = cont_scaler.fit_transform(data[columns_continuous])

# Update the categorical variables list
columns_categorical_final = [column for column in data.columns if column not in columns_continuous and column not in ['PassengerGroup']]

# Print correlation matrix
if(Verbose1):
    fig,axs = plt.subplots(1, 2, tight_layout=True, dpi=800)
    sns.heatmap(data[columns_continuous+['Transported']].corr(), linewidth=0.5, cmap='plasma', xticklabels=True, yticklabels=True, ax=axs[0], cbar_kws={'shrink':0.42})
    axs[0].set_title('Correlation matrix - Continuous variables', size='small')
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].tick_params(labelsize='x-small')
    sns.heatmap(data[columns_categorical_final].corr(), linewidth=0.5, cmap='plasma', xticklabels=True, yticklabels=True, ax=axs[1], cbar_kws={'shrink':0.42})
    axs[1].set_title('Correlation matrix - Categorical variables', size='small')
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].tick_params(labelsize='x-small')

# Save apart the target variable
data_out = data['Transported']*1
data.drop(columns='Transported',inplace=True)


#%% Perform the very same preprocessing on the test data

# Extract meaningful information from 'PassengerId','Cabin'
data_test.insert(loc=int(np.where(data_test.columns=='PassengerId')[0][0]), column='PassengerGroup', value=data_test['PassengerId'].str.split('_').str[0])
data_test.insert(loc=int(np.where(data_test.columns=='Cabin'      )[0][0]), column='CabinDeck',      value=data_test['Cabin'      ].str.split('/').str[0])
data_test.insert(loc=int(np.where(data_test.columns=='Cabin'      )[0][0]), column='CabinSide',      value=data_test['Cabin'      ].str.split('/').str[2])
data_test.insert(loc=int(np.where(data_test.columns=='Name'       )[0][0]), column='FirstName',      value=data_test['Name'       ].str.split(' ').str[0])
data_test.insert(loc=int(np.where(data_test.columns=='Name'       )[0][0]), column='LastName',       value=data_test['Name'       ].str.split(' ').str[1])
data_test_id = data_test['PassengerId']
data_test.drop(columns=['Name','PassengerId','Cabin'],inplace=True)

# For each continuous variable, sum 1 and convert to logarithmic scale
for column in columns_continuous:
    data_test[column] = np.log10(data_test[column]+1)
    
# Transform True/False categorical variables into 1/0 values
for column in columns_categorical_bool:
    data_test[column] = data_test[column].replace({True: 1, False: 0})

# Transform the other categorical variables into one-hot-encoded variables
for column in columns_categorical_nobool:
    convert_into_onehot(data_test, column, categories_labels[column]['value'])
    data_test.drop(columns=column,inplace=True)

# Scale continuous variables into [0,1] interval
data_test[columns_continuous] = cont_scaler.transform(data_test[columns_continuous])

# Check that the columns of training and test data correspond
assert np.all(data_test.columns == data.columns)


#%% Data inputing on training data

# Compute the number of NaNs before imputing
n_nan_before = find_n_nans(data)

# Columns lists
cols_expenses    = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
cols_binary      = ['CryoSleep','CabinSideS','VIP']
cols_homeplanet  = [col for col in data.columns if 'HomePlanet'   in col]
cols_deck        = [col for col in data.columns if 'CabinDeck'    in col]
cols_destination = [col for col in data.columns if 'Destination'  in col]

# Determine the age before which there are no expenses and impute consequently
max_age_no_money = data[data[cols_expenses].sum(axis=1,skipna=False) == 0]['Age'].max()
data.loc[data['Age'] <= max_age_no_money,cols_expenses] = 0

# Inspect if 'PassengerGroup','FirstName','LastName' allow to infer some categorical variables
vars_inputs = ['PassengerGroup','FirstName','LastName']
vars_groups = [cols_homeplanet, cols_deck, cols_destination] + [[col] for col in cols_binary]
if(Verbose2):
    for var_input in vars_inputs:
        count_distributions_per_group(data,var_input, vars_groups)

# 'CabinSide' and 'HomePlanet' can be imputed starting from 'PassengerGroup' and 'LastName'
for vars_group in [cols_homeplanet, ['CabinSideS']]:
    impute_category_based_on_group(data,'PassengerGroup',vars_group)
impute_category_based_on_group(data,'LastName',cols_homeplanet)

# Impute 'HomePlanet' based on 'CabinDeck' (as suggested by the contingency matrices)
data.loc[data['CabinDeckA'] == 1,cols_homeplanet] = [0,1] # 'HomePlanet' = Europe
data.loc[data['CabinDeckB'] == 1,cols_homeplanet] = [0,1] # 'HomePlanet' = Europe
data.loc[data['CabinDeckC'] == 1,cols_homeplanet] = [0,1] # 'HomePlanet' = Europe
data.loc[data['CabinDeckD'] == 1,cols_homeplanet] = [1,0] # 'HomePlanet' = Earth

# Check that people in 'CryoSleep' have no bill and impute the bill variables consequently
cryo_has_no_bill = data[data['CryoSleep']==1][cols_expenses].sum().sum() == 0
if(Verbose2):
    print('Do people in cryosleep have no bill? ', cryo_has_no_bill)
data.loc[data['CryoSleep']==1,                         cols_expenses] = 0 # CryoSleep -> bill = 0
data.loc[data[cols_expenses].sum(axis=1,skipna=False)!=0,'CryoSleep'] = 0 # bill != 0 -> !CryoSleep 
    
# Impute VIP variables based on 'HomePlanet' and 'CabinDeck'
data.loc[data['HomePlanetEarth'] == 1,'VIP'] = 0 # 'VIP' = 0
data.loc[data['CabinDeckG'     ] == 1,'VIP'] = 0 # 'VIP' = 0
    
# Compute the number of NaNs after strong imputing
n_nan_after_strong = find_n_nans(data)
 
# Perform weaker imputings to fill as many columns as possible
for column in ['VIP']+cols_expenses:
    data[column].fillna(0, inplace=True)                            # set to 0 all VIP and bill NaN records
data.loc[data[cols_destination[0]].isna(),cols_destination] = [1,0] # set the NaN destinations to the most frequent value
data.loc[data[cols_homeplanet [0]].isna(),cols_homeplanet ] = [1,0] # set the NaN homplanets   to the most frequent value

# Replace columns containing NaNs with further one-hot encoded columns
data['CryoSleepY']  = data['CryoSleep' ].replace(to_replace={np.nan:0})
data['CryoSleepN']  = data['CryoSleep' ].replace(to_replace={np.nan:0, 1:0, 0:1})
data['CabinSideSY'] = data['CabinSideS'].replace(to_replace={np.nan:0})
data['CabinSideSN'] = data['CabinSideS'].replace(to_replace={np.nan:0, 1:0, 0:1})

# Remove the irrelevant columns for the classification
data.drop(columns=['PassengerGroup','FirstName','LastName','Age','CryoSleep','CabinSideS']+cols_deck,inplace=True)

# Compute the number of NaNs after weaker imputing
n_nan_after_weak = find_n_nans(data)
nans_table  = pd.DataFrame({'Before':n_nan_before, 'After_strong':n_nan_after_strong, 'After_weak': n_nan_after_weak})
print('\nThe number of NaNs for each column of training data (before and after imputing) are:\n', nans_table, sep='')


#%% Perform the very same imputing process on the test data
data_test.loc[data_test['Age'] <= max_age_no_money,cols_expenses] = 0     # impute bill variables based on 'Age'
for vars_group in [cols_homeplanet, ['CabinSideS']]:
    impute_category_based_on_group(data_test,'PassengerGroup',vars_group) # impute 'CabinSide' and 'HomePlanet' from 'PassengerGroup' groupings
impute_category_based_on_group(data_test,'LastName',cols_homeplanet)      # impute 'HomePlanet' from 'LastName' groupings

# Impute 'HomePlanet' based on 'CabinDeck' (as suggested by the contingency matrices)
data_test.loc[data_test['CabinDeckA'] == 1,cols_homeplanet] = [0,1] # 'HomePlanet' = Europe
data_test.loc[data_test['CabinDeckB'] == 1,cols_homeplanet] = [0,1] # 'HomePlanet' = Europe
data_test.loc[data_test['CabinDeckC'] == 1,cols_homeplanet] = [0,1] # 'HomePlanet' = Europe
data_test.loc[data_test['CabinDeckD'] == 1,cols_homeplanet] = [1,0] # 'HomePlanet' = Earth

# Further strong imputings
data_test.loc[data_test['CryoSleep']==1,                         cols_expenses] = 0 # CryoSleep -> bill = 0
data_test.loc[data_test[cols_expenses].sum(axis=1,skipna=False)!=0,'CryoSleep'] = 0 # bill != 0 -> !CryoSleep 
data_test.loc[data_test['HomePlanetEarth'] == 1,'VIP'] = 0 # impute VIP based on 'HomePlanet'
data_test.loc[data_test['CabinDeckG'     ] == 1,'VIP'] = 0 # impute VIP based on 'CabinDeck'

# Weaker imputings to fill as many columns as possible
for column in ['VIP']+cols_expenses:
    data_test[column].fillna(0, inplace=True)                                 # set to 0 all VIP and bill NaN records
data_test.loc[data_test[cols_destination[0]].isna(),cols_destination] = [1,0] # set the NaN destinations to the most frequent value
data_test.loc[data_test[cols_homeplanet [0]].isna(),cols_homeplanet ] = [1,0] # set the NaN homplanets   to the most frequent value

# Replace columns containing NaNs with further one-hot encoded columns
data_test['CryoSleepY']  = data_test['CryoSleep' ].replace(to_replace={np.nan:0})
data_test['CryoSleepN']  = data_test['CryoSleep' ].replace(to_replace={np.nan:0, 1:0, 0:1})
data_test['CabinSideSY'] = data_test['CabinSideS'].replace(to_replace={np.nan:0})
data_test['CabinSideSN'] = data_test['CabinSideS'].replace(to_replace={np.nan:0, 1:0, 0:1})

# Remove the irrelevant columns for the classification
data_test.drop(columns=['PassengerGroup','FirstName','LastName','Age','CryoSleep','CabinSideS']+cols_deck,inplace=True)

# Compute the number of NaNs after weaker imputing
print('\nThe number of NaNs for each column of test data (before and after imputing) are:\n', find_n_nans(data_test), sep='')

# Check that the columns of training and test data correspond
assert np.all(data_test.columns == data.columns)


#%% Build the predictor and perform the prediction

# Convert the data DataFrame to a NumPy array
X = data    .to_numpy()
y = data_out.to_numpy().reshape((-1,1))

# Keras model instantiation and input
ann = keras.models.Sequential()
ann.add(keras.Input(shape=(X.shape[1],)))

# Keras model hidden part
ann.add(layers.Dense(units=64, activation='relu'))
ann.add(layers.Dense(units=32, activation='relu'))

# Keras model output
ann.add(layers.Dense(units=1, activation='sigmoid'))

# Model inspection
assert ann.predict(X).shape == y.shape
ann.summary()

# Model compiling and training
ann.compile(optimizer=m1_settings['optimizer'],loss=m1_settings['loss'],metrics=['Accuracy'])
ann.fit(x=X,y=y,batch_size=m1_settings['batch_size'],epochs=m1_settings['n_epochs'],verbose=1,shuffle=True)

# Model prediction
y_test = ann.predict(data_test.to_numpy()).flatten() > 0.5
prediction_table = pd.DataFrame({'PassengerId': data_test_id, 'Transported': y_test})
if(not Debug):
    prediction_filename = 'predictions_Boselli_SpaceShipTitanic_' + f'{datetime.now():%Y-%m-%d_%H-%M-%S%z}' + '.csv'
    prediction_table.to_csv(path_or_buf=os.path.join(Data_filepath,prediction_filename),index=False)