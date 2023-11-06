# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 00:37:28 2023

@author: Valentine
"""
import os
import subprocess
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np





source_dir = './data/raw_mp3'
convert_dir = './data/converted_wav'
mel_dir = './data/mels'



for genre in os.listdir(source_dir):
    dst_genre_dir = convert_dir + '/' + genre
    
    try:
        os.mkdir(f'{convert_dir}/{genre}')
    except:
        pass
    try:
        os.mkdir(f'{mel_dir}/{genre}')
    except:
        pass
    
    for file in os.listdir(f'{source_dir}/{genre}'):
        dst_file = f'{dst_genre_dir}/{file}'
        convrt_dst_file = '.' + ''.join(dst_file.split('.')[0:-1])
        
        print(f'> convert {file} to .wav')
        subprocess.call(['ffmpeg', '-i', f'{source_dir}/{genre}/{file}', convrt_dst_file + '.wav'])
        
        #Значение sr - частота дискретизации
        ampls, rate = librosa.load(convrt_dst_file + '.wav', sr=16000)
        lenrates = len(ampls)
        
        #Вырезаем 20 секунд из середины аудиозаписи
        lr = int(lenrates/2-rate*10)
        rr = int(lenrates/2+rate*10)
        
        print(f'    - write crop wav to {convrt_dst_file}')
        sf.write(convrt_dst_file + '.wav', ampls[lr:rr], rate, subtype='PCM_24')
        
        print(f'    - write mel to {convrt_dst_file}')
        mel = librosa.feature.melspectrogram(y=ampls[lr:rr], sr=rate, n_mels=160, fmin=1, fmax=8192)
        np.save(f"./data/mels/{genre}/{''.join(file.split('.'))[0:-1]}", mel)


# data = np.load('mel.npy')
# aud = librosa.feature.inverse.mfcc_to_audio(data, sr=16000, n_mels=128, fmin=1, fmax=8192)
# print(np.shape(aud))
# # sf.write('au_mel.wav', aud, 16000, subtype='PCM_24')


