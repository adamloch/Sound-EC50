import glob
import os
import numpy as np
import argparse
import pandas as pd


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


def data_split_train_valid_test(data, train=[str(x) for x in range(1, 4)],
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
    #print(data.loc[data['fold'].isin(train)])
    train = data.loc[data['fold'].isin(train)]
    test = data.loc[data['fold'].isin(test)]
    validation = data.loc[data['fold'].isin(validation)]
    
    # shuffling
    train = train.sample(frac = 1)
    test = test.sample(frac = 1)
    validation = validation.sample(frac = 1)
    return train, validation, test


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
    train_data, validation_data, test_data = data_split_train_valid_test(
        data, train, validation, test)
    return [train_data, validation_data, test_data]

def read_labels(data):
    '''
        Arguments:
        dataset

        Returns:
        Dictionary in order {id: class name}
    '''
    classes = data.category.unique().tolist()
    ids = data.target.unique().tolist()
    dictionary = dict(zip(ids,classes))

    return dictionary



def _main_(args):
    get_data_info('esc50.csv')

    


if __name__ == '__main__':
    args = False
    _main_(args)
