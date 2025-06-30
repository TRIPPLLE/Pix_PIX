

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Convolution2DTranspose,Concatenate,Dropout,Input,LeakyReLU,BatchNormalization,Activation,Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam

from keras.utils import plot_model
from keras.initializers import RandomNormal

## definign the discrimnator  accordinn to the research paper
def new_discriminator(image_Shape):
  init=RandomNormal(stddev=0.02)
  in_src_image=Input(shape=image_Shape)
  target_src_image=Input(shape=image_Shape)
  merged=Concatenate()([in_src_image,target_src_image])
  d=Conv2D(64,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(merged)
  d=LeakyReLU(alpha=0.2)(d)
  d=Conv2D(128,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)
  d=LeakyReLU(alpha=0.2)(d)
  d=BatchNormalization()(d)
  d=Conv2D(256,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)
  d=LeakyReLU(alpha=0.2)(d)
  d=BatchNormalization()(d)
  d=Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)
  d=LeakyReLU(alpha=0.2)(d)
  d=BatchNormalization()(d)
  d=Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)
  d=LeakyReLU(alpha=0.2)(d)
  d=BatchNormalization()(d)
  d=Conv2D(1,(4,4),kernel_initializer=init)(d)
  patch_acti=Activation('sigmoid')(d)
  model=Model([in_src_image,target_src_image],patch_acti)
  opt=Adam(learning_rate=0.0002,beta_1=0.5)
  model.compile(loss='binary_crossentropy',optimizer=opt,loss_weights=[0.5])
  return model

def decoder_block(layer_in,filters,skip_in,dropout=True):
  init=RandomNormal(stddev=0.02)
  g=Conv2DTranspose(filters,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(layer_in)
  if dropout:
    g=Dropout(0.5)(g,training=True)
  g=Concatenate()([g,skip_in])
  g=Activation('relu')(g)
  return g

def generator(image_shape=(256,256,3)):
  init=RandomNormal(stddev=0.02)
  in_image=Input(shape=image_shape)
  e1 = encoder_block(in_image, 64, batchnorml=False)

  e2=encoder_block(e1,128)
  e3=encoder_block(e2,256)
  e4=encoder_block(e3,512)
  e5=encoder_block(e4,512)
  e6=encoder_block(e5,512)
  e7=encoder_block(e6,512 )
  b=Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(e7)
  b=Activation('relu')(b)
  d1=decoder_block(b,512,e7)
  d2=decoder_block(d1,512,e6)
  d3=decoder_block(d2,512,e5)
  d4=decoder_block(d3,512,e4,dropout=False)
  d5=decoder_block(d4,256,e3,dropout=False)
  d6=decoder_block(d5,128,e2,dropout=False)
  d7=decoder_block(d6,64,e1,dropout=False)
  g=Conv2DTranspose(image_shape[2],(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d7)
  out_image=Activation('tanh')(g)
  model=Model(in_image,out_image)
  return model
  #defining the moel

def encoder_block(layer_in,filters,batchnorml=True):
  init=RandomNormal(stddev=0.02)
  g=Conv2D(filters,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(layer_in)
  if batchnorml:
    g=BatchNormalization()(g)
  g=LeakyReLU(0.02)(g)
  return g

gen_model=generator((256,256,3))

gen_model.summary()
plot_model(gen_model,to_file='gen_model.png',show_shapes=True,show_layer_names=True)

def Gan(image_shape,gen_model,den_model):
  for layer in den_model.layers:
    if not isinstance(layer,BatchNormalization):
      layer.trainable=False
  in_src=Input(shape=image_shape)
  gen_out=gen_model(in_src)
  den_out=den_model([in_src,gen_out])
  model=Model(in_src,[gen_out,den_out])
  opt=Adam(learning_rate=0.0002,beta_1=0.5)


  model.compile(loss=['binary_crossentropy','mae'],optimizer=opt,loss_weights=[1,100])
  return model

def generated_real_Sample(dataset, n_samples, patch_shape):
    # Take n_samples from dataset and convert to list of (input, target) pairs
    dataset_subset = list(dataset.take(n_samples).as_numpy_iterator())
    
    X = np.array([pair[0] for pair in dataset_subset])
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    
    return X, y

def fake_sampple(gen_model,samples,shape_p):
   fake=gen_model.predict(samples)
   y=np.zeros((len(fake),shape_p,shape_p,1))
   return fake,y

def performance_observance(step,gen_model,dataset,sample=3):
  [Train_A,Train_B]=generated_real_Sample(dataset,sample,1)
  Train_A = np.squeeze(Train_A, axis=1)
  Train_B = np.squeeze(Train_B, axis=1)

  gen_imgs=gen_model.predict(Train_A)
  gen_imgs=(gen_imgs+1)/2.0
  fake_A,_=fake_sampple(gen_model,Train_A,1)
  fake_A=(fake_A+1)/2.0
  Train_B=(Train_B+1)/2.0


  for i in range(sample):
    plt.subplot(2,sample,1+i)
    plt.axis('off')
    plt.imshow(Train_A[i])

  for i in range(sample):
    plt.subplot(2,sample,1+i)
    plt.axis('off')
    plt.imshow(fake_A[i])


  for i in range(sample):
    plt.subplot(2,sample,1+i)
    plt.axis('off')
    plt.imshow(Train_B[i])

  save1='plot'
  plt.savefig(save1)
  plt.close()

  model='model.keras'
  gen_model.save(model)
  print('saved the model and plot')

def Training_pix_pix(din_model,gen_model,dataset,epoch=100,ba=10):
  pach=din_model.shape[1]
  X1,X2=dataset
  batch_perPepo=int(len(X1)/pach)
  n_step=batch_perPepo*epoch
  #gen_model.summary()
  for i in range(n_step):
    [x_real,X_real],y_real=generated_real_Sample(X1,batch_perPepo,pach)
    x_fake,y_fake=fake_sampple(gen_model,x_real,pach)
    dloss=gen_model.train_on_batch(  [x_real,X_real],y_real)
    Dloss=gen_model.train_on_batch(x_real,[y_real,X_real])
    gloss=gen_model.train_on_batch(x_real,[y_real,X_real])
    if (i+1)%(batch_perPepo*10 )==0:

      performance_observance(i,gen_model,dataset)


def Training_pix2pix(disc_model, gen_model, gan_model, dataset, epochs=100, batch_size=1):
    patch_shape = disc_model.output_shape[1]
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for step, (X_realA, X_realB) in enumerate(dataset):
            # Real label
            y_real = tf.ones((batch_size, patch_shape, patch_shape, 1))

            # Generate fake image
            X_fakeB = gen_model(X_realA, training=True)
            y_fake = tf.zeros((batch_size, patch_shape, patch_shape, 1))

            # Train discriminator
            d_loss1 = disc_model.train_on_batch([X_realA, X_realB], y_real)
            d_loss2 = disc_model.train_on_batch([X_realA, X_fakeB], y_fake)

            # Train generator via GAN
            g_loss = gan_model.train_on_batch(X_realA, [X_realB,y_real])

            if (step + 1) % 100 == 0:
                print(f"Step {step+1}, D1={d_loss1:.3f}, D2={d_loss2:.3f}, G={g_loss}")
                performance_observance(step, gen_model, dataset)

image_shape = (256, 256, 3)
disc_model = new_discriminator(image_shape)
gen_model = generator(image_shape)
gan_model = Gan(image_shape, gen_model, disc_model)

# Example dataset with images (X1: input, X2: target)
# Must be preprocessed: [-1, 1] range
 # shape (num_samples, 256, 256, 3)
import tensorflow as tf
import os

def load_image_pair(input_path, target_path):
    input_image = tf.io.read_file(input_path)
    input_image = tf.image.decode_jpeg(input_image)
    input_image = tf.image.resize(input_image, [256, 256])
    input_image = (tf.cast(input_image, tf.float32) / 127.5) - 1  # Normalize to [-1, 1]

    target_image = tf.io.read_file(target_path)
    target_image = tf.image.decode_jpeg(target_image)
    target_image = tf.image.resize(target_image, [256, 256])
    target_image = (tf.cast(target_image, tf.float32) / 127.5) - 1

    return input_image, target_image

# Paths
input_dir = "train_X_2"
target_dir = "train_Y_2"

     
input_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
target_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])

dataset = tf.data.Dataset.from_tensor_slices((input_paths, target_paths))
dataset = dataset.map(load_image_pair).batch(1)

Training_pix2pix(disc_model, gen_model, gan_model, dataset, epochs=100, batch_size=1)








