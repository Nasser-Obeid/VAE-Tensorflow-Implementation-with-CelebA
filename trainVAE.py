import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

import os
import glob
import shutil
from typing import Iterable

import modelVAE

PATH : str = r"/mnt/c/Users/User/Desktop/datasets/CelebA/img_align_celeba/"
CSV_FILE : str = r"/mnt/c/Users/User/Desktop/datasets/CelebA/list_attr_celeba.csv"
IMG_FOLDER : str = r"/mnt/c/Users/User/Desktop/datasets/CelebA/img_align_celeba/img_align_celeba/"
IMG_FOLDER2 : str = r"/mnt/c/Users/User/Desktop/datasets/CelebA/img_align_celeba/"
SIZE : tuple[int] = (128, 128)
BATCH_SIZE : int = 100
ROOT_DIRECTORY : str = os.getcwd()


def preprocess(img, label=None):
    
  img = tf.io.read_file(img)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.cast(img, tf.float32)
  img = tf.image.resize(img, SIZE)
  img = img / 255.0
  img = tf.reshape(img, SIZE + (3,))

  if label != None:
    return img, label
  return img


def dataSetGeneratorFromCSV(csvfile : str, imagefolder : str, batch_size : int, n_length: int = 10000, attrib: str = 'Male'):
 
  root_dir = os.getcwd()
 
  df = pd.read_csv(csvfile,)
  final_table_columns = ['image_id', attrib]
  df = df.drop(columns=[col for col in df if col not in final_table_columns])
  
  images = []
  labels = []
  os.chdir(imagefolder)
  for i in range(0, n_length):
     images.append(os.path.join(imagefolder, df.loc[i].at['image_id']))
     labels.append(df.loc[i].at[attrib])
     
  if os.getcwd() != root_dir:
        os.chdir(root_dir) 
  
  imageData = tf.data.Dataset.from_tensor_slices((images, labels))
  imageData = imageData.map(preprocess)
  imageData = imageData.shuffle(buffer_size=1000).batch(batch_size=batch_size)
  
  if os.getcwd() != ROOT_DIRECTORY:
        os.chdir(ROOT_DIRECTORY) 
  
  return imageData


def dataSetGenerator(imagefolder : str, batch_size : int):
  location = os.chdir(imagefolder)
  
  l = []
  for folder in os.listdir(location):
      l.append(os.path.join(imagefolder, folder))

  images = []
  for i in l:
      os.chdir(i)
      files = glob.glob("*.jpg")
      for f in files:
          images.append(os.path.join(i, f))
  
  dataset = tf.data.Dataset.from_tensor_slices(images)
  dataset = dataset.map(preprocess)
  dataset = dataset.shuffle(buffer_size=1000).batch(batch_size=batch_size)
  
  if os.getcwd() != ROOT_DIRECTORY:
        os.chdir(ROOT_DIRECTORY) 
  
  return dataset


def scheduler(epoch, lr):
    if epoch > 15:
            return lr * 0.9
    else:
            return lr
        


def train(VAE , data, lr : float, batch_size : int, name : str, epochs : int = 10):

    vae = VAE
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['loss'])
    scheduler_lr = keras.callbacks.LearningRateScheduler(scheduler)
    history = vae.fit(data, batch_size=batch_size, epochs=epochs, callbacks=scheduler_lr)

    if os.getcwd() != ROOT_DIRECTORY:
        os.chdir(ROOT_DIRECTORY) 
    
    newpath = f'./attempts/{name}'
    
    if os.path.isdir(newpath):
        shutil.rmtree(newpath) 
    os.mkdir(newpath)
    
    encoder_path = os.path.join(newpath, f"encoder_model")
    decoder_path = os.path.join(newpath, f"decoder_model")
    
    os.mkdir(decoder_path)
    os.mkdir(encoder_path)
    
    vae.decoder.save(os.path.join(decoder_path, "model.keras"))
    vae.encoder.save(os.path.join(encoder_path, "model.keras"))
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Total Loss')
    plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
    plt.plot(history.history['kl_loss'], label='KL Loss')
    plt.title(f"Training History)")
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(newpath, "fig.png"))
    
    return [history.history['loss'], history.history['reconstruction_loss'], history.history['kl_loss']]
