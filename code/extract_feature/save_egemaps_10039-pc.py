#!/usr/bin/env python3
# bagus@ep.its.ac.id, 
# changelog:
# 2019-04-16: init code from avec
# 2019-07-02: modify to extract 10039 iemocap data

import numpy as np
import os
import time
import ntpath
import pickle

feature_type = 'egemaps'
exe_opensmile = '~/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'  
path_config   = '~/opensmile-2.3.0/config/'                                      
iemocap_path = '/media/bagustris/bagus/dataset/IEMOCAP_full_release/'

with open(iemocap_path+'data_collected_full.pickle', 'rb') as handle:
    data = pickle.load(handle)

if feature_type=='mfcc':
    folder_output = '../audio_features_mfcc/'  # output folder
    conf_smileconf = path_config + 'MFCC12_0_D_A.conf'  # MFCCs 0-12 with delta and acceleration coefficients
    opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsv 0 -timestampcsv 1 -headercsv 1'  # options from standard_data_output_lldonly.conf.inc
    outputoption = '-csvoutput'  # options from standard_data_output_lldonly.conf.inc
elif feature_type=='egemaps':
    folder_output = './audio_features_egemaps_10039/'  # output folder
    conf_smileconf = path_config + 'gemaps/eGeMAPSv01a.conf'  # eGeMAPS feature set
    opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1'  # options from standard_data_output.conf.inc
    outputoption = '-lldcsvoutput'  # options from standard_data_output.conf.inc

else:
    print('Error: Feature type ' + feature_type + ' unknown!')

if not os.path.exists(folder_output):
    os.mkdir(folder_output)

listfile = [id['id'] for id in data] 

for fn in listfile:
    filename = iemocap_path+'Session'+fn[4]+'/sentences/wav/'+fn[:-5]+'/'+fn+'.wav'
    instname = fn #os.path.splitext(filename)[0]
    outfilename = folder_output + instname + '.csv'
    opensmile_call = exe_opensmile + ' ' + opensmile_options + ' -inputfile ' + filename + ' ' + outputoption + ' ' + outfilename + ' -instname ' + instname + ' -output ?'  # (disabling htk output
    os.system(opensmile_call)
    time.sleep(0.01)

os.remove('smile.log')
