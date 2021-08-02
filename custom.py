import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import cv2
import time
from tensorflow.keras import layers
# from tensorflow import keras
# from tensorflow.keras import layers, Dense, Input, InputLayer, Flatten
# from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

# plt.figure(figsize=(20,20))
IMG_WIDTH=200
IMG_HEIGHT=200
img_folder=r'/home/ether/GenerativeArt/archive/abstract_art_512'
# for i in range(5):
#     file = random.choice(os.listdir(img_folder))
#     image_path= os.path.join(img_folder, file)
#     img=mpimg.imread(image_path)
#     ax=plt.subplot(1,5,i+1)
#     ax.title.set_text(file)
#     plt.imshow(img)
#     plt.show()

def create_dataset(img_folder):
   
    img_data_array= []
    class_name=[]

    for i in range(5):
        file = random.choice(os.listdir(img_folder))
        image_path= os.path.join(img_folder, file)
        image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
        # image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
        image=np.array(image)
        # image = image.reshape(image.shape[0]*3, 28, 28, 1).astype('float32')
        # image = image.astype('float32')
        # image /= 255 
        img_data_array.append(image)
    return (img_data_array)
        # print(image)



    # for file in os.listdir(os.path.join(img_folder)):
    #     image_path= os.path.join(img_folder,file)
    #     image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
    #     # print(image)
    
    #     image=np.array(image)
    #     # image = image.astype('float32')
    #     # image /= 255 
    #     img_data_array.append(image)
        # class_name.append(dir1)
    # return img_data_array, class_name

BUFFER_SIZE = 60000
BATCH_SIZE = 256

img_data = create_dataset(img_folder)
img_data = np.array(img_data)
print(img_data.shape)
train_dataset = tf.data.Dataset.from_tensor_slices(img_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(type(train_dataset))


# def make_generator_model():
#     model = tf.keras.Sequential()
#     model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Reshape((7, 7, 256)))
#     assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

#     model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
#     assert model.output_shape == (None, 7, 7, 128)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     assert model.output_shape == (None, 14, 14, 64)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
#     assert model.output_shape == (None, 28, 28, 1)

#     return model

# generator = make_generator_model()
# # print(generator)
# # noise = tf.random.normal([1, 100])
# # print(noise)
# # generated_image = generator(noise, training=False)
# # print(generated_image)
# # plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# # plt.show()

# def make_discriminator_model():
#     model = tf.keras.Sequential()
#     model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
#                                      input_shape=[28, 28, 1]))
#     model.add(layers.LeakyReLU())
#     model.add(layers.Dropout(0.3))

#     model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
#     model.add(layers.LeakyReLU())
#     model.add(layers.Dropout(0.3))

#     model.add(layers.Flatten())
#     model.add(layers.Dense(1))

#     return model

# discriminator = make_discriminator_model()
# # decision = discriminator(generated_image)
# # print (decision)

# # # This method returns a helper function to compute cross entropy loss
# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# def discriminator_loss(real_output, fake_output):
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss

# def generator_loss(fake_output):
#     return cross_entropy(tf.ones_like(fake_output), fake_output)

# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

                                
# EPOCHS = 50
# noise_dim = 100
# num_examples_to_generate = 16

# # # You will reuse this seed overtime (so it's easier)
# # # to visualize progress in the animated GIF)
# # seed = tf.random.normal([num_examples_to_generate, noise_dim])
# # print(seed)
# # # Notice the use of `tf.function`
# # # This annotation causes the function to be "compiled".
# @tf.function
# def train_step(images):
#     noise = tf.random.normal([BATCH_SIZE, noise_dim])

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#       generated_images = generator(noise, training=True)

#       real_output = discriminator(images, training=True)
#       fake_output = discriminator(generated_images, training=True)

#       gen_loss = generator_loss(fake_output)
#       disc_loss = discriminator_loss(real_output, fake_output)

#     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# def train(dataset, epochs):
#   # print(dataset)
#   # return
#   for epoch in range(epochs):
#     start = time.time()

#     for image_batch in dataset:
#       print(image_batch)
#       train_step(image_batch)

#     # Produce images for the GIF as you go
#     display.clear_output(wait=True)
#     generate_and_save_images(generator,
#                              epoch + 1,
#                              seed)

#     # Save the model every 15 epochs
#     if (epoch + 1) % 15 == 0:
#       checkpoint.save(file_prefix = checkpoint_prefix)

#     print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

#   # Generate after the final epoch
#   display.clear_output(wait=True)
#   generate_and_save_images(generator,
#                            epochs,
#                            seed)

# def generate_and_save_images(model, epoch, test_input):
#   # Notice `training` is set to False.
#   # This is so all layers run in inference mode (batchnorm).
#   predictions = model(test_input, training=False)

#   fig = plt.figure(figsize=(4, 4))

#   for i in range(predictions.shape[0]):
#       plt.subplot(4, 4, i+1)
#       plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
#       plt.axis('off')

#   plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
#   # plt.show()

# train(train_dataset, EPOCHS)

# # 
# # train_images = img_data.reshape(img_data.shape[0], 28, 28, 1).astype('float32')
# # train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
# # print(train_images)
