import tensorflow as tf
tf.__version__

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import pathlib


def load_data():


  data_dir = "/home/ether/GenerativeArt/archive/"
  data_dir = pathlib.Path(data_dir)
  # print(data_dir)

  image_count = len(list(data_dir.glob('*/*.jpg')))
  # print(image_count)

  # batch_size = 32
  img_height = 56
  img_width = 56

  # splitting the data for training

  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    # batch_size=batch_size
    )

      

  # train_ds = np.array(train_ds)
  # # print(type(train_ds))
  # for img in train_ds:
    # print(img)
  # print(type(train_ds))
  # print(train_ds.class_names)



  # doing the changes for generative art



  # splitting the data for validation

  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    # batch_size=batch_size
    )

  # class_names = train_ds.class_names
  # print(class_names)

  # standardize the data
  BUFFER_SIZE = 60000
  BATCH_SIZE = 32

  normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
  image_batch, labels_batch = next(iter(normalized_ds))
  # print(type(image_batch))
  image_batch = np.array(image_batch)
  image_batch = (image_batch - 127.5) / 127.5  # Normalize the images to [-1, 1]
  image_batch = image_batch.reshape(32, 56, 56,3).astype('float32')

  return image_batch


dataset = load_data()
print(dataset.shape)
# print(image_batch.shape)
# for i in image_batch:
#   print(i.shape)
# print(image_batch.shape)
# train_dataset = tf.data.Dataset.from_tensor_slices(image_batch).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# # print(train_dataset)
# # for element in train_dataset:
# #   print(type(element))
# # print(type(train_dataset))

# def make_generator_model():
#     model = tf.keras.Sequential()
#     model.add(layers.Dense(180*180*3, use_bias=False, input_shape=(100,)))
#     # print(model.output_shape)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Reshape((180, 180, 3)))
#     # print(model.output_shape)
#     assert model.output_shape == (None, 180, 180, 3)  # Note: None is the batch size

#     model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False))
#     # print(model.output_shape)
#     assert model.output_shape == (None, 180, 180, 3)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    
#     assert model.output_shape == (None, 180, 180, 3)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
#     print(model.output_shape)
#     assert model.output_shape == (None, 180, 180, 3)

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
#     input_shape = [256,180, 180, 3]
#     # x = tf.random.normal(input_shape)
#     model = tf.keras.Sequential()
#     model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
#                                      input_shape=input_shape))
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

# # # # This method returns a helper function to compute cross entropy loss
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
# seed = tf.random.normal([num_examples_to_generate, noise_dim])
# # print(seed)
# # # Notice the use of `tf.function`
# # # This annotation causes the function to be "compiled".
# @tf.function
# def train_step(images):
#     noise = tf.random.normal([BATCH_SIZE, noise_dim])

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#       generated_images = generator(noise, training=True)
#       # print(images)
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
#   # len_data = []
#   for epoch in range(epochs):
#     start = time.time()

#     for image_batch in dataset:
#       # print(type(image_batch))
#       # len_data.append(image_batch)
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















# # # print(type(train_dataset))
# # # train_images = image_batch.reshape(image_batch.shape[0], 28, 28, 1).astype('float32')
# # # print(image_batch.shape)
# # # first_image = image_batch[0]
# # # first_label = labels_batch[0]
# # # for image in image_batch:
# #   # print(type(image))
# # # print(first_image)
# # # print(first_label)
# # # Notice the pixels values are now in `[0,1]`.
# # # print(np.min(first_image), np.max(first_image))

# # # tuning data for performance

# # # AUTOTUNE = tf.data.AUTOTUNE

# # # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# # # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# # # num_classes = 1

# # # model = tf.keras.Sequential([
# # #   tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
# # #   tf.keras.layers.Conv2D(32, 3, activation='relu'),
# # #   tf.keras.layers.MaxPooling2D(),
# # #   tf.keras.layers.Conv2D(32, 3, activation='relu'),
# # #   tf.keras.layers.MaxPooling2D(),
# # #   tf.keras.layers.Conv2D(32, 3, activation='relu'),
# # #   tf.keras.layers.MaxPooling2D(),
# # #   tf.keras.layers.Flatten(),
# # #   tf.keras.layers.Dense(128, activation='relu'),
# # #   tf.keras.layers.Dense(num_classes)
# # # ])

# # # model.compile(
# # #   optimizer='adam',
# # #   loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
# # #   metrics=['accuracy'])

# # # model.fit(
# # #   train_ds,
# # #   validation_data=val_ds,
# # #   epochs=3
# # # )