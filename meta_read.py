import glob
import os
import numpy as np
import argparse
import pandas as pd
import sys
import subprocess

import glob
import wavio


def load_csv_config(path_csv, classes='all', fold=[x for x in range(1, 11)]):
    '''
    Arguments:
        path_csv -- path to your metadata file
        classes -- list of names which has to be included
                ex. ['dog','thunderstorm']
        fold -- list of int which corespods to included folds ex. [3,1]

    Returns:
        dataframe which contains info about files
    '''
    df = pd.read_csv(path_csv)
    del df['esc10'], df['src_file'], df['take']
    if classes != 'all':
        df = df.loc[df['category'].isin(classes)]
    df = df.loc[df['fold'].isin(fold)]

    return df


def data_split_train_valid_test(data, data_type='train', train=[str(x) for x in range(1, 4)],
                                validation=['4'], test=[5]):
    '''
    Function which split data and shuffle data within frame
    of 
        Arguments:
        path_csv -- path to your metadata file
        classes -- list of names which has to be included
                ex. ['dog','thunderstorm']
        fold -- list of int which corespods to included folds ex. [3,1]

    Returns:
        train - dataframe to train
        test - dataframe to test
        validation - dataframe to validation
    '''
    train = data.loc[data['fold'].isin(train)]
    test = data.loc[data['fold'].isin(test)]
    validation = data.loc[data['fold'].isin(validation)]

    # shuffling
    train = train.sample(frac=1)
    test = test.sample(frac=1)
    validation = validation.sample(frac=1)
    if data_type == 'train':
        return train
    elif data_type == 'test':
        return test
    else:
        return validation


def get_data_info(path_csv, classes='all', train=[x for x in range(1, 4)],
                  validation=['4'], test=['5']):
    '''
        Arguments:
        path_csv -- path to your metadata file
        classes -- list of names which has to be included
                ex. ['dog','thunderstorm']
        train, validation, test -- lists of int which corespods
                to included folds ex. [3,1]

    Returns:
        array [train_data, validation_data, test_data]
    '''
    data = load_csv_config(path_csv, classes, train+validation+test)
    classes = read_labels(data)
    data = data_split_train_valid_test(data)
    return (data)


def read_labels(data):
    '''
        Arguments:
        dataset

        Returns:
        Dictionary in order {id: class name}
    '''
    classes = data.category.unique().tolist()
    ids = data.target.unique().tolist()
    dictionary = dict(zip(ids, classes))

    return dictionary


def create_dataset(src_path, esc50_dst_path):
    print('* {} -> {}'.format(src_path, esc50_dst_path))

    esc50_dataset = {}

    for fold in range(1, 6):
        esc50_dataset['fold{}'.format(fold)] = {}
        esc50_sounds = []
        esc50_labels = []

        for wav_file in sorted(glob.glob(os.path.join(src_path, '{}-*.wav'.format(fold)))):
            sound = wavio.read(wav_file).data.T[0]
            #start = sound.nonzero()[0].min()
            #end = sound.nonzero()[0].max()
            # sound = sound[start: end + 1]  # Remove silent sections
            label = int(os.path.splitext(wav_file)[0].split('-')[-1])
            esc50_sounds.append(sound)
            esc50_labels.append(label)

        esc50_dataset['fold{}'.format(fold)]['sounds'] = esc50_sounds
        esc50_dataset['fold{}'.format(fold)]['labels'] = esc50_labels

    np.savez(esc50_dst_path, **esc50_dataset)


def _main_(args):
    # create_dataset('/home/adam/ESC-50-master/audio',
    #               '/home/adam/ESC-50-master/audio-ed')
    print(get_data_info('esc50.csv').head)


if __name__ == '__main__':
    args = False
    _main_(args)
