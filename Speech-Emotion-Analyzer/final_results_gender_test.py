#!/usr/bin/env python
# coding: utf-8

# ## Importing the required libraries

# In[1]:


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    # In[2]:


    from keras import regularizers


    # In[3]:


    import os


    # In[4]:


    mylist= os.listdir('output/')


    # In[5]:


    type(mylist)


    # In[6]:


    print(mylist[0])


    # In[7]:


    print(mylist[0][6:-16])


    # ## Plotting the audio file's waveform and its spectrogram

    # In[8]:


    data, sampling_rate = librosa.load('output/output10.wav')


    # In[9]:



    import os
    import pandas as pd
    import librosa
    import glob

    plt.figure(figsize=(15, 5))
    librosa.display.waveplot(data, sr=sampling_rate)


    # In[11]:


    import matplotlib.pyplot as plt
    import scipy.io.wavfile
    import numpy as np
    import sys
    import librosa

    #sr,x = scipy.io.wavfile.read('output/output10.wav')
    x, sr = librosa.load('output/output10.wav',sr = None)

    ## Parameters: 10ms step, 30ms window
    nstep = int(sr * 0.01)
    nwin  = int(sr * 0.03)
    nfft = nwin

    window = np.hamming(nwin)

    ## will take windows x[n1:n2].  generate
    ## and loop over n2 such that all frames
    ## fit within the waveform
    nn = range(nwin, len(x), nstep)

    X = np.zeros( (len(nn), nfft//2) )
    print(nwin)
    for i,n in enumerate(nn):
        print(str(i)+','+str(n))
        xseg = x[n-nwin:n]
        z = np.fft.fft(xseg,nfft)
        X[i,:] = np.log(np.abs(z[:nfft//2]))

    plt.imshow(X.T, interpolation='nearest',
        origin='lower',
        aspect='auto')

    plt.show()


    # ## Setting the labels

    # In[12]:


    feeling_list=[]
    for item in mylist:
        if item[6:-16]=='02' and int(item[18:-4])%2==0:
            feeling_list.append('female_calm')
        elif item[6:-16]=='02' and int(item[18:-4])%2==1:
            feeling_list.append('male_calm')
        elif item[6:-16]=='03' and int(item[18:-4])%2==0:
            feeling_list.append('female_happy')
        elif item[6:-16]=='03' and int(item[18:-4])%2==1:
            feeling_list.append('male_happy')
        elif item[6:-16]=='04' and int(item[18:-4])%2==0:
            feeling_list.append('female_sad')
        elif item[6:-16]=='04' and int(item[18:-4])%2==1:
            feeling_list.append('male_sad')
        elif item[6:-16]=='05' and int(item[18:-4])%2==0:
            feeling_list.append('female_angry')
        elif item[6:-16]=='05' and int(item[18:-4])%2==1:
            feeling_list.append('male_angry')
        elif item[6:-16]=='06' and int(item[18:-4])%2==0:
            feeling_list.append('female_fearful')
        elif item[6:-16]=='06' and int(item[18:-4])%2==1:
            feeling_list.append('male_fearful')
        elif item[:1]=='a':
            feeling_list.append('male_angry')
        elif item[:1]=='f':
            feeling_list.append('male_fearful')
        elif item[:1]=='h':
            feeling_list.append('male_happy')
        #elif item[:1]=='n':
            #feeling_list.append('neutral')
        elif item[:2]=='sa':
            feeling_list.append('male_sad')


    # In[13]:


    labels = pd.DataFrame(feeling_list)


    # In[14]:


    labels[:10]


    # ## Getting the features of audio files using librosa

    # In[15]:


    df = pd.DataFrame(columns=['feature'])
    bookmark=0
    for index,y in enumerate(mylist):
        if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08' and mylist[index][:2]!='su' and mylist[index][:1]!='n' and mylist[index][:1]!='d':
            X, sample_rate = librosa.load('RawData/'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X,
                                                sr=sample_rate,
                                                n_mfcc=13),
                            axis=0)
            feature = mfccs
            #[float(i) for i in feature]
            #feature1=feature[:135]
            df.loc[bookmark] = [feature]
            bookmark=bookmark+1


    # In[16]:


    df[:5]


    # In[17]:


    df3 = pd.DataFrame(df['feature'].values.tolist())


    # df3[:5]

    # In[18]:


    newdf = pd.concat([df3,labels], axis=1)


    # In[19]:


    rnewdf = newdf.rename(index=str, columns={"0": "label"})


    # In[20]:


    rnewdf[:5]


    # In[21]:


    from sklearn.utils import shuffle
    rnewdf = shuffle(newdf)
    rnewdf[:10]


    # In[22]:


    rnewdf=rnewdf.fillna(0)


    # ## Dividing the data into test and train

    # In[23]:


    newdf1 = np.random.rand(len(rnewdf)) < 0.8
    train = rnewdf[newdf1]
    test = rnewdf[~newdf1]


    # In[24]:


    train[250:260]


    # In[25]:


    trainfeatures = train.iloc[:, :-1]


    # In[26]:


    trainlabel = train.iloc[:, -1:]


    # In[27]:


    testfeatures = test.iloc[:, :-1]


    # In[31]:


    testlabel = test.iloc[:, -1:]


    # In[32]:


    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder

    X_train = np.array(trainfeatures)
    y_train = np.array(trainlabel)
    X_test = np.array(testfeatures)
    y_test = np.array(testlabel)

    lb = LabelEncoder()

    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))


    # In[33]:


    y_train


    # In[34]:


    X_train.shape


    # ## Changing dimension for CNN model

    # In[35]:



    x_traincnn =np.expand_dims(X_train, axis=2)
    x_testcnn= np.expand_dims(X_test, axis=2)


    # In[36]:


    model = Sequential()

    model.add(Conv1D(256, 5,padding='same',
                     input_shape=(216,1)))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5,padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    #model.add(Conv1D(128, 5,padding='same',))
    #model.add(Activation('relu'))
    #model.add(Conv1D(128, 5,padding='same',))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)


    # In[37]:


    model.summary()


    # In[38]:


    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])


    # ### Removed the whole training part for avoiding unnecessary long epochs list

    # In[39]:


    cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))


    # In[40]:


    plt.plot(cnnhistory.history['loss'])
    plt.plot(cnnhistory.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    # ## Saving the model

    # In[112]:


    model_name = 'Emotion_Voice_Detection_Model.h5'
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


    # In[133]:


    import json
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)


    # ## Loading the model

    # In[137]:


    # loading json and creating model
    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


    # ## Predicting emotions on the test data

    # In[138]:


    preds = loaded_model.predict(x_testcnn,
                             batch_size=32,
                             verbose=1)


    # In[139]:


    preds


    # In[115]:


    preds1=preds.argmax(axis=1)


    # In[116]:


    preds1


    # In[117]:


    abc = preds1.astype(int).flatten()


    # In[118]:


    predictions = (lb.inverse_transform((abc)))


    # In[119]:


    preddf = pd.DataFrame({'predictedvalues': predictions})
    preddf[:10]


    # In[120]:


    actual=y_test.argmax(axis=1)
    abc123 = actual.astype(int).flatten()
    actualvalues = (lb.inverse_transform((abc123)))


    # In[121]:


    actualdf = pd.DataFrame({'actualvalues': actualvalues})
    actualdf[:10]


    # In[122]:


    finaldf = actualdf.join(preddf)


    # ## Actual v/s Predicted emotions

    # In[1]:


    finaldf[170:180]


    # In[129]:


    finaldf.groupby('actualvalues').count()


    # In[130]:


    finaldf.groupby('predictedvalues').count()


    # In[131]:


    finaldf.to_csv('Predictions.csv', index=False)


    # ## Live Demo

    # #### The file 'output10.wav' in the next cell is the file that was recorded live using the code in AudioRecoreder notebook found in the repository

    # In[2]:


    data, sampling_rate = librosa.load('output10.wav')


    # In[486]:


    get_ipython().run_line_magic('pylab', 'inline')
    import os
    import pandas as pd
    import librosa
    import glob

    plt.figure(figsize=(15, 5))
    librosa.display.waveplot(data, sr=sampling_rate)


    # In[487]:


    #livedf= pd.DataFrame(columns=['feature'])
    X, sample_rate = librosa.load('output10.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive


    # In[488]:


    livedf2= pd.DataFrame(data=livedf2)


    # In[489]:


    livedf2 = livedf2.stack().to_frame().T


    # In[490]:


    livedf2


    # In[491]:


    twodim= np.expand_dims(livedf2, axis=2)


    # In[492]:


    livepreds = loaded_model.predict(twodim,
                             batch_size=32,
                             verbose=1)


    # In[493]:


    livepreds


    # In[494]:


    livepreds1=livepreds.argmax(axis=1)


    # In[495]:


    liveabc = livepreds1.astype(int).flatten()


    # In[496]:


    livepredictions = (lb.inverse_transform((liveabc)))
    livepredictions







