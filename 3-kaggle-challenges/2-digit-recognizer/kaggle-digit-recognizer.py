# -*- coding: utf-8 -*-
"""
Analysis of MNIST digits dataset for the Digit Recognizer Kaggle competition:
the aim is to build a classificator of handwritten digits (0,1,2,3...9), which
in this code is attained with a convolutional neural network (CNN) approach

@author: Andrea Boselli
"""

#%% Relevant libraries
import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import tensorflow        as tf
from datetime                import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import OneHotEncoder
from tensorflow              import keras
from tensorflow.keras        import layers


#%% Main settings
On_Colab         = False       # whether this code is run on Colab or not
Data_filepath    = os.getcwd() # path to the current dataset
Train_filename   = 'train.csv' # filename of the training data
Test_filename    = 'test.csv'  # filename of the test     data
n_plotted_digits = 7           # number of plotted digits in the data inspection
Verbose          = True        # verbose       mode
Debug            = False       # debug         mode
Deterministic    = True        # deterministic mode (fix all the seeds you can)


#%% CNN model settings
cnn_settings = {}
cnn_settings['optimizer']  = 'Adam'
cnn_settings['loss']       = 'CategoricalCrossentropy'
cnn_settings['batch_size'] = 32
cnn_settings['n_epochs']   =  2
cnn_settings['val_split']  =  0.1


#%% Operations to be performed in case Colab is employed
if(On_Colab):
    from google.colab import drive
    drive.mount('/content/gdrive')
    #'/content/gdrive/MyDrive/...'


#%% Load, analyse and process training data
data      = pd.read_csv(filepath_or_buffer = os.path.join(Data_filepath,Train_filename))
data_test = pd.read_csv(filepath_or_buffer = os.path.join(Data_filepath,Test_filename ))
data_no_nan      = not data     .isnull().values.any() # check the absence of nan in the training data
data_test_no_nan = not data_test.isnull().values.any() # check the absence of nan in the test     data

# Compute relevant quantites of training and test data
n_data      = len(data)                     # number of training data
n_data_test = len(data_test)                # number of test     data
img_size    = int(np.sqrt(data.shape[1]-1)) # number lines and rows in the square images

# Preprocess the input: select input columns, convert to NumPy array, reshape and rescale it
X      = data.drop(['label'], axis=1).values.reshape([n_data,      img_size, img_size, 1]) / 255. # training data
X_test = data_test                   .values.reshape([n_data_test, img_size, img_size, 1]) / 255. # test     data

# Preprocess the output: select the output column and transform with one-hot encoding
y = data['label'].values.reshape([n_data, 1])
y_labels, y_lab_counts  = np.unique(y, return_counts=True) # extract the different labels from the labels column
y_nlabels = len(y_labels)                                  # extract the number of different labels
y_encoder = OneHotEncoder(categories=[y_labels],sparse_output=False)
y_one_hot = y_encoder.fit_transform(y)

# Plot some instances of the training data
if(Verbose):
    fig, axs = plt.subplots(n_plotted_digits, y_nlabels, dpi = 800)
    for j in range(y_nlabels):
        img_idxs = np.random.choice(np.where(y.flatten()==y_labels[j])[0], size=n_plotted_digits, replace=False)
        for i in range(n_plotted_digits):
            pcm = axs[i,j].imshow(X[img_idxs[i],:,:,0], cmap='plasma',interpolation='nearest', vmin=0, vmax=1)
            axs[i,j].axis('off')
        axs[0,j].set_title(str(y_labels[j]))
    fig.suptitle('Digits from the MNIST training data')
    fig.colorbar(pcm, ax=axs)

# Plot the relative frequencies of the labels in the training data
if(Verbose):
    fig, ax = plt.subplots(1, 1, dpi = 800)
    ax.bar(x = np.arange(y_nlabels), height = y_lab_counts / n_data, 
           tick_label = y_labels, color='deepskyblue')
    ax.set_title ('Labels relative frequencies')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Relative frequencies')
    ax.grid(visible=True)
    ax.set_axisbelow(True)


#%% In case of deterministic mode, fix the seeds of the RNGs
if(Deterministic):
    tf.keras.utils.set_random_seed(0)
    split_random_state = 0
else:
    split_random_state = None


#%% Define the CNN model to be trained on the training data
cnn = keras.Sequential(name='CNN')
cnn.add(keras.Input(shape=(img_size,img_size,1),name='CNN_input'))

# Convolutional part
cnn.add(layers.Conv2D(filters=32, strides=2, kernel_size=2, activation='relu'))
cnn.add(layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(layers.Conv2D(filters=32, strides=2, kernel_size=2, activation='relu'))
cnn.add(layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(layers.Flatten(name='CNN_flatten'))

# Fully connected part
cnn.add(layers.Dense(units=32, activation='relu'))
cnn.add(layers.Dense(units=16, activation='relu'))

# Final layer and one-hot encoding
cnn.add(layers.Dense(units=y_nlabels,activation='softmax',name='CNN_softmax'))

# Model summary
cnn_summary = []
cnn.summary(line_length=80,print_fn=(lambda row: cnn_summary.append(row)))
cnn_summary = "\n".join(cnn_summary)
print(cnn_summary)


#%% Test the consistency of the CNN model, compile it and train it

# Test the consistency of the model
assert cnn.predict(X).shape == y_one_hot.shape

# Compile the model
cnn.compile(optimizer=cnn_settings['optimizer'],loss=cnn_settings['loss'],metrics=['Accuracy'])

# Randomly split into training and validation data
X_train, X_val, y_train, y_val = train_test_split(X,y_one_hot,test_size=cnn_settings['val_split'],
                                                  random_state=split_random_state)

# Train the model
print("\n\nModel training:")
cnn.fit(x=X_train,y=y_train,batch_size=cnn_settings['batch_size'],epochs=cnn_settings['n_epochs'],
        verbose=1,validation_data=(X_val,y_val),shuffle=False)


#%% Model evaluation and prediction generation

# Evaluate the final model
print("\n\nFinal model evaluation:")
cnn_evaluation = cnn.evaluate(x=X_val,y=y_val,batch_size=cnn_settings['batch_size'],
                              verbose=1,return_dict=True)

if(not Debug):
    # Generate the summary of the current script execution
    exec_summary_newlines = [cnn_summary]
    exec_summary_newlines.append("\n\n--+ Final model settings +--\n")
    for key in cnn_settings.keys():
        exec_summary_newlines.append(key+': '+cnn_settings[key].__repr__())
    exec_summary_newlines.append("\n\n--+ Final model performance +--\n")
    for key in cnn_evaluation.keys():
        exec_summary_newlines.append(key+': '+str(cnn_evaluation[key]))
    exec_summary = "\n".join(exec_summary_newlines)
    
    # Generate the predictions of the current model
    print("\n\nPredictions generation:")
    y_test_one_hot = cnn.predict(x=X_test,batch_size=cnn_settings['batch_size'],verbose=1)
    y_test         = y_labels[np.argmax(y_test_one_hot,axis=1)]
    y_id           = np.arange(1,n_data_test+1,1)
    prediction_df  = pd.DataFrame({"ImageId":y_id,"Label":y_test})
    
    # Save the outputs
    current_time = f'{datetime.now():%Y-%m-%d_%H-%M-%S%z}'
    output_folder = os.path.join(Data_filepath, 'predictions_Boselli_'+current_time)
    os.mkdir(output_folder)
    with open(os.path.join(output_folder,'model_settings_'+current_time+'.txt'), "w") as f:
        f.write(exec_summary)
    prediction_df.to_csv(path_or_buf=os.path.join(output_folder,'model_predictions_'+current_time+'.csv'),index=False)