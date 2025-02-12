import tensorflow as tf
import numpy as np
from tensorflow import keras


def encoder(latent_dim : int, ) -> tf.keras.Model:
  encoder_input = tf.keras.layers.Input(shape=(128, 128, 3,))
  x = tf.keras.layers.Conv2D(32, activation='relu',kernel_size = 1, padding='same')(encoder_input)
  x = tf.keras.layers.Conv2D(64, activation='relu', kernel_size=2 , strides=(2, 2),padding='same')(x)
  x = tf.keras.layers.Conv2D(128, activation='relu',kernel_size=2 , strides=(2, 2),padding='same')(x)
  x = tf.keras.layers.Conv2D(256, activation='relu',kernel_size=2 , strides=(2, 2),padding='same')(x)
  x = tf.keras.layers.Conv2D(512, activation='relu',kernel_size=2 , strides=(2, 2),padding='same')(x)
  x = tf.keras.layers.Flatten()(x)

  z_mean = tf.keras.layers.Dense(latent_dim)(x)
  z_log_var = tf.keras.layers.Dense(latent_dim)(x)

  z = Sampling()([z_mean, z_log_var])
  
  return tf.keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")


class Sampling(tf.keras.layers.Layer):
  def call(self, inputs):
    mean, log_var = inputs
    batch = tf.keras.ops.shape(mean)[0]
    dim = tf.keras.ops.shape(mean)[1]
    epsilon = tf.keras.random.normal(shape=(batch, dim))
    return mean + tf.keras.ops.exp(0.5 * log_var) * epsilon
  
  
def decoder(latent_dim : int) -> tf.keras.Model:
  decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
  x = tf.keras.layers.Dense(8 * 8 * 512, activation='relu')(decoder_input)
  x = tf.keras.layers.Reshape((8, 8, 512))(x)
  x = tf.keras.layers.Conv2DTranspose(256, activation='relu',kernel_size=2 , strides=(2, 2),padding='same')(x)
  x = tf.keras.layers.Conv2DTranspose(128, activation='relu',kernel_size=2 , strides=(2, 2),padding='same')(x)
  x = tf.keras.layers.Conv2DTranspose(64, activation='relu',kernel_size=2 , strides=(2, 2),padding='same')(x)
  x = tf.keras.layers.Conv2DTranspose(32, activation='relu',kernel_size=2 , strides=(2, 2),padding='same')(x)

  output_decoder = tf.keras.layers.Conv2DTranspose(3,kernel_size=1 , strides=(1, 1),padding='same')(x)
    
  return tf.keras.Model(decoder_input, output_decoder, name="decoder")


class VAE(tf.keras.Model):
  def __init__(self,encoder_model, decoder_model, **kwargs):
    super().__init__(**kwargs)
    self.encoder=encoder_model
    self.decoder=decoder_model
    self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
    self.total_loss_tracker = tf.keras.metrics.Mean(name='loss')
    self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    
    
  @property
  def metrics(self):
    return [self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
    ]


  @tf.function
  def train_step(self,x_batch):
    with tf.GradientTape() as tape:
      mean , log_var, z = self.encoder(x_batch)
      reconstruction = self.decoder(z)
      recon_loss = tf.keras.ops.mean(
        tf.keras.ops.sum(
          keras.losses.binary_crossentropy(x_batch, reconstruction),
          axis=(1, 2), 
        )
      )
      kl_loss = -0.5 * (1 + log_var - tf.keras.ops.square(mean) - tf.keras.ops.exp(log_var))
      kl_loss = tf.keras.ops.mean(tf.keras.ops.sum(kl_loss, axis=1))
      total_loss = recon_loss + kl_loss
      
    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(recon_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    
    return {
      "loss" : self.total_loss_tracker.result(),
      "reconstruction_loss" : self.reconstruction_loss_tracker.result(),
      "kl_loss": self.kl_loss_tracker.result(),
    }
