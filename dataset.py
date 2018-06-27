from torch.utils.data.dataset import Dataset
import pandas as pd
import os
import wavio
import random
import numpy as np
import torch
class Normalize(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sound):
        return sound/self.factor


class RandomGain(object):
    def __init__(self, db):
        self.db = db

    def __call__(self, sound):
        return sound*np.power(10, random.uniform(-self.db, self.db) / 20.0)

class ToTensor(object):
    def __call__(self, sound, label):
        return (torch.from_numpy(sound).unsqueeze(0).float(), torch.from_numpy(label).float())

class RandomCrop(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, sound):
        org_size = len(sound)
        start = random.randint(0, self.size)
        return sound[start: start + max(15000, start)]
        
class Padding(object):
    def __init__(self, width):
        self.width = width
    def __call__(self, sound):
        if(len(sound)<self.width):
            rand = random.randint(0, self.width-len(sound))
            return np.pad(sound, (rand, self.width-len(sound)-rand), 'constant')
        else:
            return sound

class ESC50_Dataset(Dataset):
    def __init__(self, path_csv, esc_50_path, classes='all', folds=[1, 2, 3], n_classes = 50, size = 24017):
        self.size = size
        self.normalize = Normalize(32768.0)
        self.random_gain = RandomGain(6)
        self.classes = n_classes
        self.tensor = ToTensor()
        self.random_crop = RandomCrop(self.size)
        self.pad = Padding(self.size)
        self.df = pd.read_csv(path_csv)
        classes = self.df.category.unique().tolist()
        ids = self.df.target.unique().tolist()
        self.dictionary = dict(zip(ids, classes))

        self.audio_path = esc_50_path
        del self.df['esc10'], self.df['src_file'], self.df['take']
        if classes != 'all':
            self.df = self.df.loc[self.df['category'].isin(classes)]
        self.df = self.df.loc[self.df['fold'].isin(folds)]

        del self.df['fold'], self.df['category']
        self.df = self.df.sample(frac=1)

    def __getitem__(self, index):
        sample = self.df.iloc[[index]]
        sound = wavio.read(self.audio_path+str(sample.iloc[0,0])).data.T[0]
        start = sound.nonzero()[0].min()
        end = sound.nonzero()[0].max()
        sound = sound[start: end + 1]
        
        #print(sound.shape)
        sound = self.random_crop(sound)
       # print(sound.shape)
        sound = self.pad(sound)
        #print(sound.shape)
        sound = self.random_gain(self.normalize(sound))
        #sound = np.expand_dims(sound, axis=0)
        label = np.zeros(self.classes)
        label[sample.iloc[0,1]] = 1.0
        #label = np.array([sample.iloc[0,1]], np.int32)
        lab = np.array(sample.iloc[0,1])
        sound, label = self.tensor(sound, label)
        #print(sound.shape)
        return (sound, lab)

    def __len__(self):
        return self.df.shape[0]


dataset = ESC50_Dataset('esc50.csv', '/home/adam/ESC-50-master/audio/')