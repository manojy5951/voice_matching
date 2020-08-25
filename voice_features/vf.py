
# for converting audio to mfcc
import numpy as np
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import pickle
def calculate_delta(array):
    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

#convert audio to mfcc features
def extract_features(audio,rate):
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True, nfft=1103)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)

    #combining both mfcc features and delta
    combined = np.hstack((mfcc_feat,delta))
    return mfcc_feat

def train_gmm(features,path):
    gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
    gmm.fit(features)
    # saving the trained gaussian model
    pickle.dump(gmm, open(path, 'wb'))
    print(' added successfully')


def load_file(FILENAME):

    sr,audio = read(FILENAME)
    features = extract_features(audio,sr)
    # train_gmm(features,FILENAME.replace('.wav','.gmm'))
    return features