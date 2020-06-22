# extrat_34_hfs: extract mean and std from LLD of 34 features
# better to save when extracting LLD

import numpy as np
feat_pad = np.load('../data/feat_34_float.npy')

feat_pad_float = feat_pad.astype(feat_pad)

feat_pad_float[feat_pad_float==0] =np.nan

# compute mean and std
mean = np.nanmean(feat_pad_float, axis=1)
std = np.nanstd(feat_pad_float, axis=1)

# output feature
feat_hfs = np.hstack([mean, std])
feat_hst = feat_hfs.reshape(10039, 1, 68)
np.save('../data/feat_hfs.npy', feat_hfs)
