import os
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def ruta_df():
    image_dir = Path('/home/Enfermedad/')

    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.PNG')) + list(image_dir.glob(r'**/*.jpeg')) + list(image_dir.glob(r'**/*.JPEG'))

    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    image_df = pd.concat([filepaths, labels], axis=1)
    
    return image_df

def obtener_train_test(image_df):
    xtrain, xtest, ytrain, ytest  = train_test_split(image_df['Filepath'], image_df['Label'], test_size=0.2, shuffle=True, stratify=image_df['Label'])
    train_df = pd.concat([xtrain, ytrain], axis=1)
    test_df = pd.concat([xtest, ytest], axis=1)
    return train_df, test_df

def input_correspondiente(input):
    generator = ImageDataGenerator(
        preprocessing_function=input
    )
    return generator

def obtener_datos_entrenar(generator, datos):
    
    datos_images = generator.flow_from_dataframe(
        dataframe=datos,
        x_col='Filepath',
        y_col='Label',
        target_size=(493, 493),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=None,
        subset='training'
    )
    return datos_images