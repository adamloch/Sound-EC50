import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_image(file_name):
    X, sample_rate = librosa.load(file_name)


def parse_audio_files(main_dir, sub_dirs, file_ext="*.wav"):
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            except Exception as e:
                print "Error encountered while parsing file: ", fn
                continue
