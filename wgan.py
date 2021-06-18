from __future__ import print_function, division

from reloading import reloading

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling1D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, ZeroPadding1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv1D, UpSampling1D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np
from database_gen import database
from sklearn import preprocessing

X,Y=database()



class WGAN():
    def __init__(self):
        self.img_rows = 16
        self.img_cols = 2560
        self.img_shape = (self.img_rows, self.img_cols)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def build_generator(self):

        model = Sequential()
        model.add(Dense(20, input_dim=self.latent_dim))
        model.add(Activation("relu"))
        model.add(Reshape((-1,1)))
        model.add(Conv1D(512, kernel_size=3, padding="same", activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(MaxPooling1D())
        model.add(Conv1D(64,kernel_size=3, padding="same", activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(12, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(5, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)


    def build_critic(self):

        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(20))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((-1,1)))
        model.add(Conv1D(32, 2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='tanh'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Load the dataset
        lim=int(len(X)*0.8)
        X_train = X.reshape([-1, 16, 2560])
        xa=[]
        xb=[]
        for i in range(len(X_train)):
            if Y[i]==[1,0]:
                xa.append(X_train[i])
            else:
                xb.append(X_train[i])

        X_train =np.array(xa)
        
        # Rescale -1 to 1
        xc=[]
        for i in X_train:
            f=preprocessing.normalize(i)
            xc.append(f)
        X_train =np.array(xc)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx].reshape([-1,16, 2560])
                
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            #print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, 1 - d_loss[0], 100*d_loss[1], 1 - g_loss[0]))
        

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            np.save('gen_imgs_normal.npy',gen_imgs)
            self.generator.save("wgan_model_normal.h5")



if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=40000, batch_size=100, sample_interval=50)
