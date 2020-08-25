import os
import argparse
import wget
import speech_recognition as sr
from voice_features import vf
import glob
import pickle
from scipy.io.wavfile import read
import numpy as np
import sys
from datetime import datetime
from scipy.spatial import distance

def log_file(text):
    return text
    # with open('logs.txt','a') as file:
    #     file.write(text)
    #     file.write(str(datetime.now()))
    #     file.write('/n ------------------------')

log_file(sys.argv[1])
log_file(sys.argv[2])
log_file(sys.argv[3])

link = sys.argv[1]
user_id = sys.argv[2]
type_use = int(sys.argv[3])



# parser = argparse.ArgumentParser(description='Voice Authentication Module')
#
# parser.add_argument('-a', dest='audio_link',
#                     default='https://api.twilio.com/2010-04-01/Accounts/AC2a78c08cd74db2b3b2a789606bd2630f/Recordings/REf32d3f42244d0a9ede5828344e120a87',
#                     help='Audio Link')
#
# parser.add_argument('-i', dest='user_id', default=127, type=int,
#
#                     help='user unique ID')
# parser.add_argument('-t', dest='operation_type', default=2, type=int,
#                     help='select operation type, 1 for enrollment , 2 for authentication')

# DEFAULT_PATH = 'User_data'
DEFAULT_PATH = 'D:\\Dayuser\\Projects\\python_voice_matching\\Voice_matching\\User_data'
r = sr.Recognizer()


def speech_to_text(audio_path):
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    text = ''
    text = r.recognize_sphinx(audio)
    # print(text)

    # perform speech to text here
    return text





def write_file_to_disk(content, path, filename,hard_update):
    log_file(content)
    log_file(path)
    log_file(filename)

    path = os.path.join(path, filename)
    if not os.path.isfile(path):

        with open(path, 'w') as file:
            file.write(content)
        file.close()
    if os.path.isfile(path) and hard_update:
        with open(path, 'w') as file:
            file.write(content)
        file.close()

def enroll_user(audio, userid):
    if os.path.isdir(os.path.join(DEFAULT_PATH, str(userid))):
        pass
    else:
        os.mkdir(os.path.join(DEFAULT_PATH, str(userid)))
    if os.path.isfile(os.path.join(DEFAULT_PATH, str(userid), str(userid) + '.wav')):
        pass
    else:

        wget.download(audio, os.path.join(DEFAULT_PATH, str(userid), str(userid) + '.wav'))
        features = vf.load_file(os.path.join(DEFAULT_PATH, str(userid), str(userid) + '.wav'))
        np.save(os.path.join(DEFAULT_PATH, str(userid), str(userid)),features)


        train_model = vf.train_gmm(features,os.path.join(DEFAULT_PATH, str(userid), str(userid) + '.wav').replace('.wav','.gmm'))
        text = speech_to_text(os.path.join(DEFAULT_PATH, str(userid), str(userid) + '.wav'))
        write_file_to_disk(text, os.path.join(DEFAULT_PATH, str(userid)), str(userid) + '.txt', False)
    # write text to file in case of enrollment
    print('Enrollment Done Successfully')


def EUDistance(d, c):
    # np.shape(d)[0] = np.shape(c)[0]
    n = np.shape(d)[0]
    p = np.shape(c)[0]
    distance = np.empty((n, p))

    if n < p:
        for i in range(n):
            copies = np.transpose(np.tile(d[:, i], (p, 1)))
            distance[i, :] = np.sum((copies - c) ** 2, 0)
    else:
        for i in range(p):
            copies = np.transpose(np.tile(c[:, i], (n, 1)))
            distance[:, i] = np.transpose(np.sum((d - copies) ** 2, 0))

    distance = np.sqrt(distance)
    return distance
def minDistance(features, codebooks):
    speaker = 0
    distmin = np.inf

    D = EUDistance(features, codebooks)
    dist = np.sum(np.min(D, axis=1)) / (np.shape(D)[0])
    # print(dist)

    return dist

def authenticate_user(audio, userId):
    gmm_files = glob.glob(DEFAULT_PATH+'/'+userId +'/**/*.gmm',
                          recursive=True)

    # features_train = np.load(os.path.join(DEFAULT_PATH,str(userId),userId+'.npy'))
    # print(gmm_files)
    # log_file(' '.join(gmm_files))
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [os.path.basename(fname).split('.gmm')[0]for fname
                in gmm_files]
    # log_file(' '.join(speakers))
    if os.path.isfile(os.path.join(DEFAULT_PATH, 'input.wav')):
        os.remove(os.path.join(DEFAULT_PATH, 'input.wav'))
    wget.download(audio, os.path.join(DEFAULT_PATH, 'input.wav'))
    # read test file
    FILENAME = os.path.join(DEFAULT_PATH, 'input.wav')
    # sr, audio = read(FILENAME)

    # extract mfcc features
    vector = vf.load_file(FILENAME)
    # if features_train.shape == vector.shape:
    # d = distance.euclidean(features_train.ravel(), np.resize(vector,features_train.shape).ravel())
    # print('|||||||',d)
    # minDistance(features_train,vector)
    # else:
    #     vector_Temp = np.resize(vector,features_train.shape)
    #     minDistance(features_train,vector_Temp)
    log_likelihood = np.zeros(len(models))

    # checking with each model one by one
    # sum_list = []
    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    pred = np.argmax(log_likelihood)
    # print(log_likelihood)
    # print('-------')
    identity = speakers[pred]
    # print((log_likelihood[pred]))
    # print(identity)
    # if voice not recognized than terminate the process
    if identity == 'unknown':
        # print("Not Recognized! Try again...")
        return

    if identity == userId and abs(log_likelihood[pred])<30.0:
        print(True)
    else:
        print(False)

    # print("Recognized as - ", identity)



if __name__ == '__main__':

    # results = parser.parse_args()
    if type_use == 1:
        print("Enrollment Started")
        log_file('enrollment')
        enroll_user(link, user_id)
    if type_use == 2:
        print('Authentication Started')
        log_file('Authentication')
        authenticate_user(link, user_id)


