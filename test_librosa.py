import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import time
from tqdm import tqdm

PATH = os.getcwd()


class data_preparation:

    def __init__(self, title):
        # self.path = path
        self.title = title

    def dir_creation(self):
        """creates the folders with the labels for the dataset"""

        path = 'examples\\{}\\'.format(self.title)

        if not os.path.isdir(path):
            os.makedirs(path)
            print('The {} created'.format(path))
'''
class visual_store:
    def __init__(self, filename, name, title):
        self.title = title
        self.filename = filename
        self.name = name

    def visual_spectrogram(self):
        """It will create the spectogram for each
        sound clip and save it as jpg format"""
        print('name', self.filename)
        plt.interactive(False)
        clip, sample_rate = librosa.core.load(self.filename, sr=None)

        fig = plt.figure(figsize=[2, 0.75])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

        '' additional features to generate augmented dataset?(added n_mfcc =40 and scaled)''
        mfccs = librosa.feature.mfcc(y=clip, sr=sample_rate, n_mfcc=128, hop_length=280, power=2.0)
        mfccsscaled = np.mean(mfccs.T, axis=0)
        # size = np.ndarray(shape=(128, 345))
        S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)

        '' additional features to generate augmented dataset (change the np.max to np.mean to avoid empty channels)''

        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

        if os.path.exists("./examples"):
            pass
        else:
            os.mkdir(os.path.join(PATH, "examples"))

        locs = 'examples\\{}\\'.format(self.title) + self.name + '.jpg'
        print(locs)
        fig.canvas.draw()

        # Now we can save it to a numpy array.

        plt.savefig(locs, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close()
        fig.clf()
        plt.close(fig)
        plt.close('all')
        del locs, name, clip, sample_rate, fig, ax, S
'''

global data
data = []

global mfc,chr,me,ton,lab,features

[mfc, chr, ton, me, lab, features] = [], [], [], [], [], []
# chr=[]
# me=[]
# ton=[]
# lab=[]
# features=[]

class dataGenerator:

    def __init__(self, filename, name, title):
        self.filename = filename
        self.name = name
        self.title = title

    def numpy_spectrogram(self):
        """It will create the spectogram for each
        sound clip and store it into a list"""
        clip, sample_rate = librosa.load(self.filename, res_type='kaiser_fast')
        mf = np.mean(librosa.feature.mfcc(y=clip, sr=sample_rate).T,axis=0)
        mfc.append(mf)

        # S = librosa.feature.melspectrogram(y=clip, sr=sample_rate, S=size)  # , S=size

        try:
            t = np.mean(librosa.feature.tonnetz(
                y=librosa.effects.harmonic(clip),
                sr=sample_rate).T, axis=0)
            ton.append(t)
        except:
            print(self.filename)
        S = np.mean(librosa.feature.melspectrogram(y=clip, sr=sample_rate).T, axis=0)
        me.append(S)
        s = np.abs(librosa.stft(clip))
        c = np.mean(librosa.feature.chroma_stft(S=s, sr=sample_rate).T, axis=0)
        chr.append(c)
        return me, mfc, ton, chr

df = pd.read_csv(r'UrbanSound8K\metadata\UrbanSound8K.csv')
# print(df.head(10))

start_time = time.time()

'''Read the .csv file from the dataset folder'''




for index, row in tqdm(df.iterrows()):
    file_name = os.path.join(os.path.abspath(r'UrbanSound8K\audio'),
                             "fold" + str(row['fold']), str(row["slice_file_name"]))
    ident = str(row['class'])
    name = file_name.split('\\')[-1].split('.')[0]
    # print(file_name)
    # print(ident)
    dc = data_preparation(ident)
    dc.dir_creation()
    # visual_prep = visual_store(file_name, name, ident)
    # visual_prep.visual_spectrogram()

    db = dataGenerator(file_name, name, ident)
    me, mfc, ton, chr = db.numpy_spectrogram()
    # lst = numpy_spectrogram(file_name, name, ident)

# with open('spectrograms.pkl', 'wb') as f:
#     pickle.dump(lst, f)
for i in tqdm(range(len(ton))):
    features.append(np.concatenate((me[i], mfc[i], ton[i], chr[i]), axis=0))

print(features[1])
print(len(features))

print('The dataset is generated and saved !!')
print("--- %s seconds ---" % round((time.time() - start_time), 2))

# data_preparation.dir_creation()
