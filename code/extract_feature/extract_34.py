# extract_34.py: extract 34 acoustic features from IEMOCAP dataset

from pyAudioAnalysis import audioBasicIO 
from pyAudioAnalysis import audioFeatureExtraction
import glob
import os
from keras.preprocessing import sequence

data_path = '/media/bagus/data01/dataset/IEMOCAP_full_release/'

# this list of wav files is consistent with labels
# checked with == operator (data_id == files_id)
files = glob.glob(os.path.join(data_path + './Session?/sentences/wav/*/', 
                               '*.wav'))
files.sort()

feat = []

for f in files:
    print("Process..., ", f)
    [Fs, x] = audioBasicIO.readAudioFile(f)
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.025*Fs, 
                                                            0.010*Fs)
    feat.append(F.transpose())

feat_pad = sequence.pad_sequences(feat, dtype='float64)
np.save('../data/feat_34.npy', feat_pad)

