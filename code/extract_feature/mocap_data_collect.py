#!#/usr/bin/env python3
import numpy as np
import os
import sys

import wave
import copy
import math

from helper import *
from scipy.io import wavfile

code_path = os.path.dirname(os.path.realpath(os.getcwd()))
emotions_used = np.array(['ang', 'exc', 'neu', 'sad', 'xxx', 'fru', 'hap', 'sur', 'dis', 'fea', 'oth'])
data_path = "/media/bagus/data01/dataset/IEMOCAP_full_release/"
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000


def read_iemocap_mocap():
    data = []
    ids = {}
    for session in sessions:
        #path_to_sentences = data_path + session + '/sentences/wav/'
        path_to_wav = data_path + session + '/dialog/wav/'
        path_to_emotions = data_path + session + '/dialog/EmoEvaluation/'
        path_to_transcriptions = data_path + session + '/dialog/transcriptions/'
#        path_to_mocap_hand = data_path + session + '/dialog/MOCAP_hand/'
#        path_to_mocap_rot = data_path + session + '/dialog/MOCAP_rotated/'
#        path_to_mocap_head = data_path + session + '/dialog/MOCAP_head/'
 
        files2 = os.listdir(path_to_wav)
        files = []
        for f in files2:
            if f.endswith(".wav"):
                if f[0] == '.':
                    files.append(f[2:-4])
                else:
                    files.append(f[:-4])

        for f in files:
            print(f)
            mocap_f = f
            if (f=='Ses05M_script01_1b'):
                mocap_f = 'Ses05M_script01_1' 
            
            #wav = get_audio(path_to_wav, f + '.wav')
            transcriptions = get_transcriptions(path_to_transcriptions, f + '.txt')
            emotions = get_emotions(path_to_emotions, f + '.txt')
            #sample = split_wav(wav, emotions)

            for ie, e in enumerate(emotions):
                '''if 'F' in e['id']:
                    e['signal'] = sample[ie]['left']
                else:
                    e['signal'] = sample[ie]['right']'''

                #e['signal'] = get_audio(path_to_sentences + f[:-5] + '/', f + '.wav')
                #_, e['signal'] = wavfile.read(path_to_sentences + f[:-5] + '/' + f + '.wav')
                e['transcription'] = transcriptions[e['id']]
                #e['mocap_hand'] = get_mocap_hand(path_to_mocap_hand, mocap_f + '.txt', e['start'], e['end'])
                #e['mocap_rot'] = get_mocap_rot(path_to_mocap_rot, mocap_f + '.txt', e['start'], e['end'])
                #e['mocap_head'] = get_mocap_head(path_to_mocap_head, mocap_f + '.txt', e['start'], e['end'])
                if e['emotion'] in emotions_used:
                    if e['id'] not in ids:
                        data.append(e)
                        ids[e['id']] = 1

    sort_key = get_field(data, "id")
    return np.array(data)[np.argsort(sort_key)]

data = read_iemocap_mocap()

import pickle
with open(data_path + 'data_collected_10039.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
