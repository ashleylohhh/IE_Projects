#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.

# # Deep Conditional Convolutional Generative Adversarial Network

# In[1]:


import tensorflow as tf


# In[2]:


tf.__version__


# In[3]:


# To generate GIFs
get_ipython().system('pip install -q imageio')


# In[4]:


import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
from tensorflow.keras import Model
import time


from IPython import display


# In[5]:


from keras import backend as K

from keras.layers import Input, Dense, Reshape, Flatten, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.utils import to_categorical


# ### Load and prepare the dataset
# 
# You will use the MNIST dataset to train the generator and the discriminator. The generator will generate handwritten digits resembling the MNIST data.

# In[6]:


import tensorflow_datasets as tfds
import tensorflow as tf


# In[7]:


tfds.list_builders()
builder = tfds.builder("fashion_mnist")
builder.download_and_prepare()


# In[8]:


print(builder.info)


# In[9]:


(train_dataset_raw) = builder.as_dataset(split="train", as_supervised=True)


# In[10]:


def format_example(image, label):
  image = tf.cast(image, tf.float32)
  # scale from [0,255] to [-1,1]
  image = (image - 127.5) / 127.5
  image = tf.image.resize(image, (28, 28))
  return image, label


# In[11]:


train_dataset = train_dataset_raw.map(format_example)


# In[12]:


i = 0
for image, label in train_dataset.take(49):
  # define subplot
  plt.subplot(7, 7, 1 + i)
  # turn off axis
  plt.axis('off')
  # plot raw pixel data
  image = (image + 1.0) / 2.0
  plt.imshow(image[:,:,0], cmap='gray')
  i = i + 1
  print(label)
plt.show()


# ## Create the models
# 
# Both the generator and discriminator are defined using the [Keras Sequential API](https://www.tensorflow.org/guide/keras#sequential_model).

# ### The Generator
# 
# The generator uses `tf.keras.layers.Conv2DTranspose` (upsampling) layers to produce an image from a seed (random noise). Start with a `Dense` layer that takes this seed as input, then upsample several times until you reach the desired image size of 28x28x1. Notice the `tf.keras.layers.LeakyReLU` activation for each layer, except the output layer which uses tanh.

# With kernels of size 5, we use the sequential API and afterwards the functional API to combine the labels
# 

# In[13]:


z_dim = 100
def make_generator_model(n_classes):
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,))) # https://stackoverflow.com/questions/56081975/output-dimension-of-reshape-layer
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    z = Input(shape=(z_dim, ))

    # Conditioning label
    label = Input(shape=(1,), dtype='int32')
    
    # embedding layer:
    # turns labels into dense vectors of size z_dim
    # produces 3D tensor with shape: (batch_size, 1, z_dim)
    label_embedding = Embedding(n_classes, z_dim, input_length=1)(label)
    
    # Flatten the embedding 3D tensor into 2D  tensor with shape: (batch_size, z_dim)
    label_embedding = Flatten()(label_embedding)
    
    # Element-wise product of the vectors z and the label embeddings
    joined_representation = multiply([z, label_embedding])
    
    img = model(joined_representation)
    model = Model([z, label], img)
    return model


# Use the (as yet untrained) generator to create an image.

# In[14]:


n_classes = 10
generator = make_generator_model(n_classes)
tf.keras.utils.plot_model(generator, show_shapes=True, to_file='generator.png')


# In[15]:


#generator.summary()
noise = tf.random.normal([1, 100])
from numpy.random import randint
noisy_label = randint(0, n_classes, 1)
generated_image = generator([noise, noisy_label], training=False)
plt.imshow((generated_image[0, :, :, 0] + 1) / 2, cmap='gray')


# ### The Discriminator
# 
# The discriminator is a CNN-based image classifier.

# In[16]:


img_shape = [28, 28, 1]
def make_discriminator_model(n_classes):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 2]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.add(Activation('sigmoid'))

    img = Input(shape=img_shape)
    
    label = Input(shape=(1,), dtype='int32')
    
    # embedding layer:
    # turns labels into dense vectors of size 28*28*1
    # produces 3D tensor with shape: (batch_size, 1, 28*28*1)
    label_embedding = Embedding(input_dim=n_classes, output_dim=np.prod(img_shape), input_length=1)(label)
    # Flatten the embedding 3D tensor into 2D  tensor with shape: (batch_size, 28*28*1)
    label_embedding = Flatten()(label_embedding)
    # Reshape label embeddings to have same dimensions as input images
    label_embedding = Reshape(img_shape)(label_embedding)
    
    # concatenate images with corresponding label embeddings
    concatenated = Concatenate(axis=-1)([img, label_embedding])
    
    prediction = model(concatenated)
    
    model =  Model([img, label], prediction)
    return model


#  Functional API with kernel 3x3 and activation function

# In[17]:


def define_discriminator(in_shape=(28,28,1), n_classes=10):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model


# In[18]:


n_classes = 10
discriminator = make_discriminator_model(n_classes)
#tf.keras.utils.plot_model(discriminator, show_shapes=True, to_file='discriminator.png')


# Use the (as yet untrained) discriminator to classify the generated images as real or fake. The model will be trained to output positive values for real images, and negative values for fake images.

# In[19]:


#discriminator.summary()
from numpy.random import randint
label = randint(0, n_classes, 1)
print(label)
decision = discriminator([generated_image, label])
print (decision)


# ## Define the loss and optimizers
# 
# Define loss functions and optimizers for both models.
# 

# In[20]:


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# ### Discriminator loss
# 
# This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.

# In[21]:


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# ### Generator loss
# The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, we will compare the discriminators decisions on the generated images to an array of 1s.

# In[22]:


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# The discriminator and the generator optimizers are different since we will train two networks separately.

# In[23]:


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
metricsAcc=tf.keras.metrics.BinaryAccuracy()


# ### Save checkpoints
# This notebook also demonstrates how to save and restore models, which can be helpful in case a long running training task is interrupted.

# In[24]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt2")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# ## Define the training loop
# 

# In[25]:


noise_dim = 100
num_examples_to_generate = 100

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
from numpy import asarray
seed_images = tf.random.normal([num_examples_to_generate, noise_dim])
seed_labels = asarray([x for _ in range(10) for x in range(10)])
print(seed_labels)


# The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.

# In[26]:


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, labels):
    noise_image = tf.random.normal([BATCH_SIZE, noise_dim])
    noise_labels = label = randint(0, n_classes, BATCH_SIZE)
    #print("coming into the function")
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

      real_output = discriminator([images, labels], training=True)
      #print("salida con imagenes reales {:.5f}".format(tf.math.reduce_mean(real_output)))
      generated_images = generator([noise_image, noise_labels], training=True)
      fake_output = discriminator([generated_images, noise_labels], training=True)
      #print("salida con imagenes falsas {:.5f}".format(tf.math.reduce_mean(fake_output)))
      
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
      
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# In[27]:


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    j = 0
    for image_batch, label_batch in dataset:
      #print(j)
      train_step(image_batch, label_batch)
      #j = j + 1

    print("A NEW EPOCH")
    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator, epoch + 1, seed_images, seed_labels)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator, epochs, seed_images, seed_labels)


# **Generate and save images**
# 

# In[28]:


def generate_and_save_images(model, epoch, test_input_images, test_input_labels):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model([test_input_images, test_input_labels], training=False)

  fig = plt.figure(figsize=(10,10))

  for i in range(predictions.shape[0]):
      plt.subplot(10, 10, i+1)
      # scale [-1, 1] to [0, 1]
      image = (predictions[i, :, :, 0] + 1.0) / 2.0
      plt.imshow(image, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


# ## Train the model
# Call the `train()` method defined above to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).
# 
# At the beginning of the training, the generated images look like random noise. As training progresses, the generated digits will look increasingly real. After about 50 epochs, they resemble MNIST digits. This may take about one minute / epoch with the default settings on Colab.

# In[29]:


BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 100
train_dataset_shuffled = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train(train_dataset_shuffled, EPOCHS)


# Restore the latest checkpoint.

# In[30]:


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# ## Create a GIF
# 

# In[31]:


# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


# In[32]:


display_image(EPOCHS)


# Use `imageio` to create an animated gif using the images saved during training.

# In[33]:


anim_file = 'dcgan_3b.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import IPython
if IPython.version_info > (6,2,0,''):
  display.Image(filename=anim_file)


# If you're working in Colab you can download the animation with the code below:

# In[34]:


try:
  from google.colab import files
except ImportError:
   pass
else:
  files.download(anim_file)


# In[34]:




