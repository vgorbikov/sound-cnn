

import subprocess
import librosa
import soundfile as sf
import tensorflow as tf
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import random
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_datasets(create_test_set: bool = False):
    ds_dir = './data/mels'
    genres = os.listdir(ds_dir)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    y_test_names = []
    
    for genre in genres:
        genre_dir = f'{ds_dir}/{genre}'
        print(f'=== Load genre: {genre} ===')
        tracks = os.listdir(genre_dir)
        for track_num in range(len(tracks)):
            if (create_test_set) and (track_num % 4) ==  0:
                print(f'    - Load track: {tracks[track_num]} as test')
                y_test.append(genre)
                y_test_names.append(tracks[track_num])
                x_test.append(np.load(f'{genre_dir}/{tracks[track_num]}'))
            else:
                print(f'    - Load track: {tracks[track_num]} as train')
                y_train.append(genre)
                x_train.append(np.load(f'{genre_dir}/{tracks[track_num]}'))
            
            
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), np.array(y_test_names)
        
    
x_train, y_train, x_test, y_test, y_test_names = get_datasets()

print(f'Загружено треков для обучения: {np.shape(y_train)}')
print(f'Загружено треков для тестирования: {np.shape(y_test)}')
print(f'Размер входных тензоров: {np.shape(x_train[0])}')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Посмотрим входные данные, выберем случайную пару
rand_show = random.randint(0, len(y_train))

fig, ax = plt.subplots()
ax.imshow(x_train[rand_show])
ax.set_title(f"Mel-спектрограмма трека №{rand_show}; Жанр: {y_train[rand_show]}")



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def generate_categories_codes(y):
    categories = np.unique(y)
    categories_count = len(categories)
    categories_codes = {categories[i]:np.zeros(categories_count, dtype=int) for i in range(categories_count)}
    for i in range(categories_count):
        categories_codes[categories[i]][i] = 1
    return categories_codes



def genres_to_categories(y, categories_codes):
    y_cat = []
    for yi in y:
        y_cat.append(categories_codes[yi])
    return np.array(y_cat)



#Разбиваем по категориям
categories_codes = generate_categories_codes(y_train)
y_train_cat = genres_to_categories(y_train, categories_codes)
y_test_cat = genres_to_categories(y_test, categories_codes)

#Расширяем размерность входных данных, добавляем одиночный канал
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3) if np.shape(x_test) != (0,) else x_test



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inp_shape = np.shape(x_train[0])

sg_model = keras.Sequential([
    Conv2D(32, (6, 6), padding='valid', activation='relu', input_shape=inp_shape),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3,  activation='softmax')
    ])

print(sg_model.summary())
print(np.shape(y_train_cat))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sg_model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
 
his = sg_model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



