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
from datetime                import datetime
from sklearn.ensemble        import RandomForestClassifier
from sklearn.experimental    import enable_iterative_imputer
from sklearn.impute          import IterativeImputer
from sklearn.linear_model    import LinearRegression
from sklearn.preprocessing   import MinMaxScaler
from sklearn.preprocessing   import OneHotEncoder


#%% Main settings
Data_filepath  = os.getcwd()    # path to the current dataset
Train_filename = 'train.csv'    # filename of the training data
Test_filename  = 'test.csv'     # filename of the test     data
Method_name    = 'RandomForest' # name of the employed method
Debug          = False          # debug mode
Verbose1       = False          # print the analysis outputs
Verbose2       = False          # plot the analysis figures
Verbose3       = False          # plot the contingency matrices
Verbose4       = False          # plot the pairplots
Verbose5       = False          # print the imputing outputs
Verbose6       = False          # print the inference outputs


#%% Settings for each employed method
if(Method_name == 'RandomForest'):
    rf_settings = {}
    rf_settings['random_state'] = 0 # 0 for deterministic, None otherwise


#%% User-defined functions

def find_n_nans(data):
    """
    For each column of the input DataFrame, return the total number of nans
    """
    return data.isna().sum()

def count_values_per_id(data, id_col, value_cols):
    """
    For each value of id_col in data DataFrame, count the number of rows having
    that id_col value, but distinct values in the value_cols columns; return the
    absolute frequencies of these counts.
    """
    nlabels_per_group = data[[id_col]+value_cols].dropna().drop_duplicates().groupby(id_col).size()
    return nlabels_per_group.value_counts()

def impute_value_based_on_id(data_in, id_col, value_col): # it works, but for sure there exists a more elegant approach
    """
    In data_in DataFrame, impute the values in column value_col based on the
    id_col column, where value_col is a unique categorical column. In particular,
    if some values in value_col are missing, they are searched among the records
    with same id_col; in case of multiple possible values or absence of such records,
    no value is imputed. Return the imputed frame, without modifying the original one
    """
    data = data_in.copy()
    lookup = data[[id_col,value_col]].dropna().drop_duplicates().drop_duplicates(subset=id_col,keep=False) # lookup table
    table_nans = data.loc[data[value_col].isna(), id_col]                                                  # rows with missing values
    table_imputed = pd.merge(table_nans,lookup, how='left', on=id_col).set_index(table_nans.index).dropna()# set missing value from lookup
    data.loc[table_imputed.index, value_col] = table_imputed[value_col]                                    # assign values
    return data.copy()


#%% Load, analyse and process data

# Load data
train = pd.read_csv(filepath_or_buffer = os.path.join(Data_filepath,Train_filename))
test  = pd.read_csv(filepath_or_buffer = os.path.join(Data_filepath,Test_filename ))

# Number of samples
n_train = len(train) # number of training samples
n_test  = len(test ) # number of test     samples

# Fractions of nan entries for all columns
nan_frac_train = find_n_nans(train) / n_train
nan_frac_test  = find_n_nans(test ) / n_test
if(Verbose1):
    print('The fraction of nans for each column of the training data is:\n',nan_frac_train,'\n',sep='')
    print('The fraction of nans for each column of the test     data is:\n',nan_frac_test, '\n',sep='')

# Many  rows  contain  missing values, both in the training and test  test;  it  is
# very unlikely that new data will be added to the analysis,; thus, it's reasonable
# to  merge  train  and  test sets for the analysis, the data  imputation  and  the
# preprocessing phases
data = pd.concat([train,test], ignore_index=True).sort_values(by='PassengerId').reset_index(drop=True) # merge train and test set

# Extract many relevant columns
data['Destination'   ] = data['Destination'].str.slice(stop=2)     # reduce characters for 'Destination'
data['CabinDeck'     ] = data['Cabin'      ].str.split('/').str[0] # cabin deck      from 'Cabin'
data['CabinSide'     ] = data['Cabin'      ].str.split('/').str[2] # cabin side      from 'Cabin'
data['FirstName'     ] = data['Name'       ].str.split(' ').str[0] # first name      from 'Name'
data['LastName'      ] = data['Name'       ].str.split(' ').str[1] # last name       from 'Name'
data['PassengerGroup'] = pd.to_numeric(data['PassengerId'].str.split('_').str[0]) # passenger group from 'PassengerId'
data['CabinNum'      ] = pd.to_numeric(data['Cabin'      ].str.split('/').str[1]) # cabin number    from 'Cabin'
data.drop(columns=['Name','Cabin'],inplace=True)                                  # 'Name' and 'Cabin' can be deleted

# Useful lists of data columns
cols_categorical = ['HomePlanet','CryoSleep','Destination','VIP','CabinDeck','CabinSide','Transported'] # categorical variables
cols_continuous  = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']                      # continuous  variables
cols_identifiers = ['PassengerId','PassengerGroup','FirstName','LastName','CabinNum']                   # (mostly) categorical variables with many categories

# Extract the labels of the categorical variables
categories_labels = {}
for col in cols_categorical:
    categories_labels[col] = data[col].value_counts().to_frame().reset_index().rename(columns={'index':'label', col:'frequency'})
    if(Verbose1):
        print("The labels of '"+col+"' column are: \n", categories_labels[col], '\n',sep='')

# Countplots of the categorical variables
if(Verbose2):
    curr_cols = list(set(cols_categorical)-{'Transported'})
    sns.set_theme(style="whitegrid")
    fig,axs = plt.subplots(1,len(curr_cols), figsize=(4*len(curr_cols), 4), dpi=800)
    for i in range(len(curr_cols)):
        sns.countplot(data=data, x=curr_cols[i], hue='Transported', palette='tab10', ax=axs[i])
    fig.suptitle('Absolute frequences of categorical variables')
    fig.tight_layout()

# Get the proportions of zeros in all the continuous variables
curr_cols = list(set(cols_continuous)-{'Age'})
cols_zero_indicators = [col+'IsZero' for col in curr_cols]
for col in curr_cols:
    data[col+'IsZero'] = (data[col] == 0)            # True if zero value, False otherwise
    data.loc[data[col].isna(),col+'IsZero'] = np.nan # set nan when necessary

# Countplots of zeros occurrences of the continuous variables
if(Verbose2):
    curr_cols = cols_zero_indicators
    sns.set_theme(style="whitegrid")
    fig,axs = plt.subplots(1,len(curr_cols), figsize=(4*len(curr_cols), 4), dpi=800)
    for i in range(len(curr_cols)):
        sns.countplot(data=data, x=curr_cols[i], hue='Transported', palette='tab10', ax=axs[i])
    fig.suptitle('Occurrences of zero in the continuous variables')
    fig.tight_layout()

# Histograms of the continuous variables
if(Verbose2):
    curr_cols = cols_continuous
    sns.set_theme(style="whitegrid")
    fig,axs = plt.subplots(1,len(curr_cols), figsize=(4*len(curr_cols), 4), dpi=800)
    for i in range(len(curr_cols)):
        sns.histplot(data=data[data[curr_cols[i]]>0], x=curr_cols[i], hue='Transported', kde=True, log_scale=True, palette='tab10', ax=axs[i])
    fig.suptitle('Histograms of continuous variables, after log transformations and excluding zero values')
    fig.tight_layout()

# Heatmaps of the frequencies of the occurrences of 2 categorical variables (normalized by row)
if(Verbose3):
    curr_cols = cols_categorical
    fig, axs = plt.subplots(len(curr_cols),len(curr_cols), figsize=(4*len(curr_cols),4*len(curr_cols)), dpi=800)
    fig.tight_layout()
    for i in range(len(curr_cols)):
        for j in range(len(curr_cols)):

            # Contingency matrix
            if(i != j):

                # Create contingency matrix
                cont_matrix = pd.crosstab(data[curr_cols[i]], data[curr_cols[j]], normalize='index', margins=False)

                # Plot contingency matrix
                sns.heatmap(cont_matrix, cmap='plasma', annot=True, cbar=False, ax = axs[i,j], annot_kws={'fontsize':'xx-small', 'rotation':45})
                axs[i,j].set_aspect('equal', adjustable='box')

            # Variable name
            if(i == j):
                axs[i,j].annotate(curr_cols[i], xy=(0.5, 0.5), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', fontsize='xx-large')
                axs[i,j].axis('off')

# Convert the bill continuous variables to logaritmic scale
curr_cols = list(set(cols_continuous)-{'Age'})
for col in curr_cols:
    data[col] = np.log10(data[col]+1) # perform log-transformation

# Scale all continuous variables into [0,1] interval
curr_cols = cols_continuous
cont_scaler = MinMaxScaler().fit(data.loc[data['PassengerId'].isin(train['PassengerId']), curr_cols]) # fit into training data
data[curr_cols] = cont_scaler.transform(data[curr_cols])

# Pairplots of the continuous variables, for each label of the categorical variables
if(Verbose4):
    curr_cols = cols_categorical
    for col in curr_cols:
        pp = sns.pairplot(data=data, hue=col, palette='tab10', vars=cols_continuous,
                          corner=True, dropna=True, kind='scatter', diag_kind='hist',
                          plot_kws = dict(marker='.', alpha = 0.3, edgecolors='none'))
        pp.fig.suptitle("Pairplot of continuous variables, for each '"+col+"' value",fontsize=32)
        pp.fig.set_dpi(150)

# Histogram of age of people with positive bill
if(Verbose2):
    curr_cols = list(set(cols_continuous)-{'Age'})
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(dpi=800)
    sns.histplot(data = data[data[curr_cols].sum(axis=1,skipna=False) > 0], x='Age', hue='Transported', kde=True, palette='tab10', ax=ax)
    ax.set_title('Age with positive bill')

# Inspect if 'PassengerGroup','FirstName','LastName','CabinNum' allow to infer some categorical variables
if(Verbose5):
    curr_cols_id    = list(set(cols_identifiers)-{'PassengerId'})
    curr_cols_categ = cols_categorical
    for col_id in curr_cols_id:
        for col_categ in curr_cols_categ:
            abs_freqs = count_values_per_id(data=data, id_col=col_id, value_cols=[col_categ])
            print("For id column '"+col_id+"', the numbers of different values of category column '"+col_categ+"' are:\n",abs_freqs,'\n', sep='')

# Check that people in 'CryoSleep' have no bill
curr_cols = list(set(cols_continuous)-{'Age'})
cryo_has_no_bill = data.loc[data['CryoSleep']==True, curr_cols].sum().sum() == 0
if(Verbose5):
    print('Do people in cryosleep have zero bill? ', cryo_has_no_bill, '\n')

# Scatterplot of 'PassengerGroup' and 'CabinNum'
if(Verbose2):
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1,2, figsize=(4*2,4), dpi=800)
    sns.scatterplot(data=data, x='PassengerGroup', y='CabinNum', hue='CabinDeck', palette='tab10', edgecolor='none', s=1, ax=axs[0])
    sns.scatterplot(data=data, x='PassengerGroup', y='CabinNum', hue='CabinSide', palette='tab10', edgecolor='none', s=1, ax=axs[1])
    fig.suptitle('CabinDeck VS PassengerGroup')
    fig.tight_layout()

# Retrieve the distribution of people in each cabin
deck_max_CabinNum = {}
deck_people_distr = {}
deck_nums_empty   = {}
max_people_per_cabin = 0
for deck in categories_labels['CabinDeck']['label']:
    deck_max_CabinNum[deck] = int(data.loc[data['CabinDeck']==deck, 'CabinNum'].max())                                                   # maximal CabinNum
    deck_people_distr[deck] = pd.crosstab(data.loc[data['CabinDeck']==deck, 'CabinNum' ],data.loc[data['CabinDeck']==deck, 'CabinSide']) # people distribution
    deck_nums_empty  [deck] = set(range(deck_max_CabinNum[deck]+1))-set(deck_people_distr[deck].index)                                   # empty cabin numbers
    max_people_per_cabin = max(max_people_per_cabin, deck_people_distr[deck].max().max())                                                # maximal number of people in a cabin

# Plot the distribution of people in each cabin
if(Verbose2):
    fig, axs = plt.subplots(1,len(categories_labels['CabinDeck']['label']), figsize=(4*len(categories_labels['CabinDeck']['label']),4), dpi=800)
    index=0
    for deck in categories_labels['CabinDeck']['label']:
        sns.heatmap(deck_people_distr[deck], cmap='rainbow', vmin=0, vmax=max_people_per_cabin, ax = axs[index], cbar=True)
        axs[index].set_title('Deck '+deck)
        index+=1
    fig.suptitle('Distribution of people in each cabin')
    fig.tight_layout()

# Inspect the relation between the available ID variables
if(Verbose5):
    curr_cols = list(set(cols_identifiers)-{'PassengerId'})
    for col1 in curr_cols:
        for col2 in list(set(curr_cols)-{col1}):
            abs_freqs = count_values_per_id(data, col1, [col2])
            print("For id column '"+col1+"', the numbers of different values of category column '"+col2+"' are:\n",abs_freqs,'\n', sep='')

# Add 'GroupSize' and 'GroupAlone' as features
counts_by_group = data[['PassengerGroup','PassengerId']].groupby('PassengerGroup').count().reset_index(drop=False) # number of passengers per group
counts_by_id    = pd.merge(left=data[['PassengerGroup']], right=counts_by_group, how='left')                       # number of passengers, assigned to each id
data['GroupSize' ] = counts_by_id['PassengerId'].copy()
data['GroupAlone'] = (data['GroupSize'] == 1)

# Retrieve the zero_bill indicator (retaining NaNs)
curr_cols = list(set(cols_continuous)-{'Age'})
zerobill = (data[curr_cols].sum(axis=1) == 0)
zerobill[(data[curr_cols].sum(axis=1) == 0) & data[curr_cols].sum(axis=1,skipna=False).isna()] = np.nan

# Inspect the relation between the bill absence and 'CryoSleep'
cryo_vs_zerobill = pd.crosstab(data['CryoSleep'], zerobill)
if(Verbose5):
    print("The relation between the bill absence and 'CryoSleep' is: \n", cryo_vs_zerobill, '\n')

#%% Strong data imputing

# Compute the number of NaNs before imputing
n_nan_before = find_n_nans(data)

# Determine the age before which there are no expenses and impute consequently
curr_cols = list(set(cols_continuous)-{'Age'})
max_age_no_bill = data.loc[data[curr_cols].sum(axis=1,skipna=False) > 0,'Age'].min() # works also on log-transformed columns
data.loc[data['Age'] < max_age_no_bill,curr_cols           ] = 0    # impute bill variables
data.loc[data['Age'] < max_age_no_bill,cols_zero_indicators] = True # impute zero-bill indicators

# Impute 'HomePlanet' from 'CabinDeck', as suggested by the contingency matrices
data.loc[data['CabinDeck'].isin(['A','B','C','T']),'HomePlanet'] = 'Europa'
data.loc[data['CabinDeck'].isin(['G'            ]),'HomePlanet'] = 'Earth'

# Impute 'HomePlanet' based on 'PassengerGroup' and 'LastName'
data = impute_value_based_on_id(data,'PassengerGroup','HomePlanet')
data = impute_value_based_on_id(data,'LastName',      'HomePlanet')

# Impute 'CabinSide' based on 'PassengerGroup'
data = impute_value_based_on_id(data,'PassengerGroup','CabinSide')

# Impute values considering that people in 'CryoSleep' have no bill
curr_cols = list(set(cols_continuous)-{'Age'})
data.loc[data['CryoSleep']==True, curr_cols           ] = 0      # CryoSleep -> bill = 0
data.loc[data['CryoSleep']==True, cols_zero_indicators] = True   # CryoSleep -> bill = 0
data.loc[data[curr_cols].sum(axis=1)>0,'CryoSleep']     = False  # bill > 0  -> !CryoSleep

# Compute the number of NaNs after strong imputing
n_nan_after_strong = find_n_nans(data)


#%% Weak data imputing

# Impute all NaN 'VIP' equal to False
data['VIP'].fillna(False, inplace=True)

# Impute 'HomePlanet' based on 'VIP'
data.loc[(data['VIP']==True ) & data['HomePlanet'].isna(),'HomePlanet'] = "Europa"
data.loc[(data['VIP']==False) & data['HomePlanet'].isna(),'HomePlanet'] = "Earth"  # when dealing with categorical data, be always precise with categories names

# Impute all NaN 'Destination' equal to 'TR'
data['Destination'].fillna('TR', inplace=True)

# Impute 'LastName' based on 'PassengerGroup'
data = impute_value_based_on_id(data,'PassengerGroup','LastName')

# Impute 'CabinDeck' based on 'PassengerGroup'
data = impute_value_based_on_id(data,'PassengerGroup','CabinDeck')

# Impute 'CabinDeck' based on 'HomePlanet'
data.loc[(data['HomePlanet']=="Mars") & data['CabinDeck'].isna(),'CabinDeck'] = 'F'

# Set dummy categories for 'CabinDeck' and 'CabinSide' that couldn't be imputed
data['CabinDeck'].fillna('U', inplace=True)
data['CabinSide'].fillna('U', inplace=True)

# Set 'CabinNum' based on 'CabinDeck' and 'CabinSide'
for deck in categories_labels['CabinDeck']['label']:
    for side in ['U','P','S']:
        sides_for_fit = [side] if (side in ['P','S']) else ['P','S']                                    # fit on both sides in case of unknown side
        fit_condition = (data['CabinDeck']==deck) & (data['CabinSide']==side) & data['CabinNum'].isna() # records that can be imputed at the current iteration
        if(Debug):
            print("Current deck '"+deck+"' - Current side '"+side+"'")                 # debug outputs
            print(data.loc[fit_condition, ['CabinDeck','CabinSide','CabinNum'] ],'\n') # imputable rows

        if(fit_condition.sum() > 0):
            line_train_data = data.loc[(data['CabinDeck']==deck) & data['CabinSide'].isin(sides_for_fit) & ~data['CabinNum'].isna(), ['PassengerGroup','CabinNum']] # training data
            regr_line = LinearRegression().fit(line_train_data[['PassengerGroup']], line_train_data[['CabinNum']])                                                  # fitted line
            predicted_nums = regr_line.predict(data.loc[fit_condition,['PassengerGroup']]).round().astype(int)                                                      # predicted values
            data.loc[fit_condition, ['CabinNum'] ] = predicted_nums

# Impute Cryosleep based on bill variables
curr_cols = list(set(cols_continuous)-{'Age'})
data.loc[(data[curr_cols].sum(axis=1,skipna=False) == 0) & data['CryoSleep'].isna(), 'CryoSleep'] = True # bill == 0  -> CryoSleep
data['CryoSleep'].fillna(False, inplace=True)                                                            # mode for the remaining records

# Impute bill variables and the consequent booleans (TODO: improve with a more sophisticated method)
curr_cols = list(set(cols_continuous)-{'Age'})
bill_imputer = IterativeImputer(random_state=0).fit(data[curr_cols]) # train the imputer
data[curr_cols] = bill_imputer.transform(data[curr_cols])            # impute the continuous variables
for col in curr_cols:
    data.loc[data[col+'IsZero'].isna(), col+'IsZero'] = (data.loc[data[col+'IsZero'].isna(), col] == 0) # impute the zero-value indicators
if(Verbose5):
    print("Does the imputing of bill variables violate the [0,1] range? ", bool(((data[curr_cols]>1) | (data[curr_cols]<0)).sum().sum()), '\n')


# Compute the number of NaNs after weaker imputing and print imputing status
n_nan_after_weak = find_n_nans(data)
nans_table  = pd.DataFrame({'Before':n_nan_before, 'After_strong':n_nan_after_strong, 'After_weak': n_nan_after_weak})
if(Verbose5):
    print('The number of NaNs for each column of training data are:\n',nans_table,'\n', sep='')


#%% Data preparation for the learning process

# Delete hopeless columns
curr_cols = ['FirstName','LastName','PassengerGroup']
data.drop(columns=curr_cols,inplace=True)

# Convert suitable categorical variables to 1/0 variables (if there remains some NaNs, it returns error)
curr_cols = ['CryoSleep','VIP','ShoppingMallIsZero','VRDeckIsZero','FoodCourtIsZero','RoomServiceIsZero','SpaIsZero','GroupAlone']
data[curr_cols] = data[curr_cols].astype(int)

# Convert suitable categorical variables to one-hot-encoded variables (NaNs should not be present)
curr_cols = ['HomePlanet','Destination','CabinDeck','CabinSide']
onehot_encoders = {}
for col in curr_cols:
    
    # Perform conversion
    onehot_encoder = OneHotEncoder(sparse=False).fit(data[[col]])       # (sparse option deprecated in the latest version)
    col_transform = pd.DataFrame(onehot_encoder.transform(data[[col]])) # one-hot encoding
    col_transform.columns = onehot_encoder.get_feature_names_out()      # set suitable names for the transformation
    
    # Store the results
    data[col_transform.columns] = col_transform # save one-hot encoded data
    data.drop(columns=col,inplace=True)         # remove original column
    onehot_encoders[col] = onehot_encoder       # save one-hot encoder

# Scale the continuous variables which have not been scaled yet
curr_cols = ['CabinNum','GroupSize']
further_cont_scalers = {}
for col in curr_cols:
    further_cont_scaler = MinMaxScaler().fit(data.loc[data['PassengerId'].isin(train['PassengerId']),[col]]) # fit into training data
    data[col] = further_cont_scaler.transform(data[[col]])                                                   # perform scaling
    further_cont_scalers[col] = further_cont_scaler                                                          # store scaler

# Remove the columns that couldn't be fully imputed
curr_cols = ['Age','CabinNum']
data.drop(columns=curr_cols,inplace=True)

# Split training and test set, properly handle the 'Transported' column
if(Verbose6):
    print('Are training and test data monotonically increasing? ', train['PassengerId'].is_monotonic_increasing, 
          ',',                                                     test ['PassengerId'].is_monotonic_increasing, '\n') # check monotonicity
X_train = data.loc[data['PassengerId'].isin(train['PassengerId'])].copy() # ID, input, output of train
X_test  = data.loc[data['PassengerId'].isin(test ['PassengerId'])].copy() # ID, input, output of test
y_train = X_train['Transported'].astype(int).to_numpy(copy=True)          # output of train
I_train = X_train['PassengerId'].to_numpy(copy=True)                      # ID of train
I_test  = X_test ['PassengerId'].to_numpy(copy=True)                      # ID of test
X_train.drop(columns=['PassengerId','Transported'],inplace=True)          # input of train
X_test.drop (columns=['PassengerId','Transported'],inplace=True)          # input of test


#%% Build the predictor and perform the prediction
if(Method_name == 'RandomForest'):
    rf_classifier = RandomForestClassifier(random_state=rf_settings['random_state']).fit(X_train, y_train) # fit the rf classifier
    y_test = rf_classifier.predict(X_test)                                                                 # predict on test data


#%% Generate the .csv with the predictions
y_test = y_test.astype(bool)                                                   # convert 1/0 to boolean
prediction_table = pd.DataFrame({'PassengerId': I_test, 'Transported': y_test})# build the prediction table
if(not Debug):
    prediction_filename = 'Predictions_SpaceShipTitanic_Boselli_' + Method_name + '_' + f'{datetime.now():%Y-%m-%d_%H-%M-%S%z}' + '.csv'
    prediction_table.to_csv(path_or_buf=os.path.join(Data_filepath,prediction_filename),index=False)


#%% Performed imputations

# AGE -> BILL VARIABLES
# CABIN_DECK -> HOME_PLANET
# PASSENGER_GROUP, LAST_NAME -> HOME_PLANET
# PASSENGER_GROUP -> CABIN_SIDE
# BILL VARIABLES <-> CRYO_SLEEP
# -----------------------------------------
# VIP ALL FALSE
# VIP -> HOMEPLANET
# DESTINATION ALL TR
# PASSENGER_GROUP -> LAST_NAME
# PASSENGER_GROUP -> CABIN_DECK
# HOMEPLANET -> CABIN_DECK
# CABINDECK, CABINSIDE TO U
# CABINDECK, CABINSIDE -> CABIN_NUN
# BILL VARIABLES -> CRYO_SLEEP
# BILL_VARIABLES 