import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def generate_mfccs(file_path, file_name, class_id):
    X, sample_rate = librosa.load(file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T,axis=0)
    print(mfccs)


def _main_(args):
    ex = generate_mfccs('','1-137-A-32.wav',1)


if __name__ == '__main__':
    args = False
    _main_(args)