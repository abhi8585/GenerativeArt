from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets
import numpy as np

from gan_utils import build_discriminator
from gan_utils import build_generator
from gan_utils import sample_images

from load import load_data




def train(generator=None,discriminator=None,gan_model=None,
          epochs=1000, batch_size=128, sample_interval=50,
          z_dim=100):
    # Load MNIST train samples
    # (X_train, _), (_, _) = datasets.mnist.load_data()

    # importing the dataset

    X_train = load_data()
    # print(X_train.shape)
    # return
    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1
    print(X_train.shape)
    # print(X_train.shape)
    # return
    # Prepare GAN output labels
    real_y = np.ones((batch_size, 1))
    fake_y = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # train disriminator
        # pick random real samples from X_train
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        
        real_imgs = X_train[idx]
        # print(real_imgs)
        # return
        # pick random noise samples (z) from a normal distribution
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        # use generator model to generate output samples
        fake_imgs = generator.predict(noise)

        # calculate discriminator loss on real samples
        disc_loss_real = discriminator.train_on_batch(real_imgs, real_y)
        
        # calculate discriminator loss on fake samples
        disc_loss_fake = discriminator.train_on_batch(fake_imgs, fake_y)
        
        # overall discriminator loss
        discriminator_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)
        
        #train generator
        # pick random noise samples (z) from a normal distribution
        noise = np.random.normal(0, 1, (batch_size, z_dim))

        # use trained discriminator to improve generator
        gen_loss = gan_model.train_on_batch(noise, real_y)

        # training updates
        print ("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, 
                                                                                   discriminator_loss[0], 
                                                                                   100*discriminator_loss[1], 
                                                                                   gen_loss))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            sample_images(epoch,generator)

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0002, 0.5),
                      metrics=['accuracy'])

generator=build_generator()

z_dim = 100
z = Input(shape=(z_dim,))
img = generator(z)

# Fix the discriminator
discriminator.trainable = False

# Get discriminator output
validity = discriminator(img)

# Stack discriminator on top of generator
gan_model = Model(z, validity)
gan_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))
gan_model.summary()

train(generator,discriminator,gan_model,epochs=30000, batch_size=32, sample_interval=100)