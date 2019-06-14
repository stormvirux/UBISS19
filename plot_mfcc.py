import numpy
import librosa
import matplotlib.pyplot as plot
import sklearn
from sklearn.model_selection import train_test_split
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
import graphviz 
#%matplotlib inline

batch_size = 2
epochs = 1000
clip_length_seconds = 120
downsampling_rate = 25
split_ratio = 0.8
random_state = random.randint(1,101)
print("Random state: " + str(random_state))
count_files = 10
y = numpy.flip(numpy.array([1, 2, 1, 2, 2, 0, 0, 2, 1, 0, 0, 2, 2, 2]))

y = y[:count_files,]

X_vectors = []
y_vectors = []

for i in [4,12]:
    filename = 'vid' + str(i) + '.wav'
    # filename = 'download.wav'
    
    wave, sr = librosa.load(filename, mono=True, sr=None)
    print(sr)
    print(wave.shape)
    #print('Sample rate: ' + str(sr))    
    #print('Wave shape for i = ' + str(i) + ': ' + str(wave.shape))
    
    for j in range(0, wave.shape[0] / clip_length_seconds / sr):
        start = j * clip_length_seconds * sr
        end = (j + 1) * clip_length_seconds * sr
        
        wave_clip = wave[start:end]
        #print('Clip shape for j = ' + str(j) + ': ' + str(wave_clip.shape))

        wave_downsampled = wave_clip[::downsampling_rate]
        #print('Downsample shape for j = ' + str(j) + ': ' + str(wave_downsampled.shape))

        mfcc = librosa.feature.mfcc(wave_downsampled, sr)
        #print('Padded mfcc shape for j = ' + str(j) + ': ' + str(mfcc.shape))
    
        X_vectors.append(mfcc)
        y_vectors.append(y[i - 1])
    
        plot.subplot(312)
        mfcc -= (numpy.mean(mfcc,axis=0) + 1e-8)
        plot.imshow(mfcc.T, cmap=plot.cm.jet, aspect='auto')
        # plot.xticks(numpy.arange(0, (mfcc.T).shape[1], int((mfcc.T).shape[1] / 6)), ['0s', '0.5s', '1s', '1.5s','2.5s','3s','3.5'])
        ax = plot.gca()
        ax.invert_yaxis()
        plot.title('the Normalized MFCC spectrum image for video ' + str(i))
        plot.show()
