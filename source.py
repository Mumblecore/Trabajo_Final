import numpy as np
import pandas as pd
import os
import librosa
import wave
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *
from keras.optimizer_v1 import rmsprop

from google.colab import drive
drive.mount('/content/drive')

def extract_mfcc(wav_file_name):
  y, sr = librosa.load(wav_file_name)
  mfccs = np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
  return mfccs

radvess_speech_labels = []
radvess_speech_data = []
for dirname, _, filenames in os.walk('/content/drive/MyDrive/IA_Final/data_set'):
  for filename in filenames:
    radvess_speech_labels.append(int(filename[7:8])-1)
    wav_file_name = os.path.join(dirname,filename)
    radvess_speech_data.append(extract_mfcc(wav_file_name))

print("carga completa")

radvess_speech_data

ravdess_speech_data_array = np.asarray(radvess_speech_data)
ravdess_speech_label_array = np.array(radvess_speech_labels)
ravdess_speech_label_array.shape
labels_categorical = to_categorical(ravdess_speech_label_array)
labels_categorical.shape

ravdess_speech_data_array.shape

x_train,x_test,y_train,y_test= train_test_split(np.array(ravdess_speech_data_array),labels_categorical, test_size =0.20,random_state=9)

number_of_samples = ravdess_speech_data_array.shape[0]
training_samples = int(number_of_samples * 0.8)
validation_samples = int(number_of_samples * 0.1)
test_samples = int(number_of_samples * 0.1)

def create_model_LSTM():
  model = Sequential()
  model.add(LSTM(128,return_sequences = False, input_shape = (40,1)))
  model.add(Dense(64))
  model.add(Dropout(0.4))
  model.add(Activation('relu'))
  model.add(Dense(32))
  model.add(Dropout(0.4))
  model.add(Activation('relu'))
  model.add(Dense(8))
  model.add(Activation('Softmax'))

  model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics = ['acuracy']) 
  return model

w = np.expand_dims(ravdess_speech_data_array[:training_samples],-1)

model_A = create_model_LSTM()
history = model_A.fit(np.expand_dims(ravdess_speech_data_array[:training_samples],-1),labels_categorical[:training_samples], validation_data = (np.expand_dims(ravdess_speech_data_array[training_samples:training_samples + validation_samples],-1), labels_categorical[training_samples:training_samples + validation_samples]),epochs=130,shuffle=True)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs= range(1, len(loss) + 1)
plt.plot(epochs,loss,'ro', label='Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title('Training validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs,acc,'ro',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

model_A.evaluate(np.expand_dims(ravdess_speech_data_array[training_samples + validation_samples:],-1),labels_categorical[training_samples + validation_samples:])

emotions ={1:'neutral',2:'calmado', 3:'feliz', 4:'triste',5:'enojado',6:'temeroso',7:'disgustado',8:'sorprendido'}
def predict (wav_filepath):
  test_point=extract_mfcc(wav_filepath)
  test_point=np.reshape(test_point,newshape=(1,40,1))
  predictions=model_A.predict(test_point)
  print(emotions[np.argmax(predictions[0])+1])
  
predict('/content/drive/MyDrive/IA_Final/test.wav')
