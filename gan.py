import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, ZeroPadding2D, Input, Add
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Activation


# Specify the path to the folder containing the images in Google Drive
folder_path = '/content/drive/MyDrive/angioml/imageAngio.zip (Unzipped Files)'
image_dimensions = (128, 128) 

image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

df = pd.DataFrame({
    'filename': image_files,
    'class': ['angiography' for _ in image_files] 
})

datagen = ImageDataGenerator(preprocessing_function=lambda x: (x/127.5)-1) 

data = datagen.flow_from_dataframe(df,
                                   folder_path,
                                   x_col='filename',
                                   y_col='class',
                                   target_size=image_dimensions,
                                   color_mode='grayscale',
                                   class_mode=None,
                                   batch_size=128)

noise_dim = 100

def build_generator():
    model = Sequential()

    model.add(Dense(128*32*32, activation="relu", input_dim=noise_dim))
    model.add(Reshape((32, 32, 128)))

    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', use_bias=False))
    model.add(InstanceNormalization())
    model.add(Activation("relu"))
    skip1 = model.output  

    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', use_bias=False))
    model.add(InstanceNormalization())
    model.add(Activation("relu"))
    skip2 = model.output 

    model.add(Conv2D(1, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(noise_dim,))
    img = model(noise)
    return Model(noise, img)

def build_discriminator():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(128,128,1), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(InstanceNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(InstanceNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(InstanceNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=(128,128,1))
    validity = model(img)
    return Model(img, validity)



def train(data, epochs, batch_size=128, save_interval=50):
    # Build and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=Adam(0.0002, 0.5),
                          metrics=['accuracy'])

    # Build the generator
    generator = build_generator()

    # The generator takes noise as input and generates imgs
    z = Input(shape=(noise_dim,))
    img = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    validity = discriminator(img)

    # The combined model (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    for epoch in range(epochs):
        # Display the progress of the training
        print(f"Current epoch: {epoch+1}/{epochs}")

        # ---------------------
        #  Train Discriminator
        # ---------------------
        imgs = data.next()  # Use .next() to get the next batch of images

        if imgs.shape[0] != batch_size:
            continue

        noise = np.random.normal(0, 1, (batch_size, noise_dim))

        # Generate a half batch of new images
        gen_imgs = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, noise_dim))

        # Train the generator (to have the discriminator label samples as valid)
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

        # If at save interval => save generated image samples and print the progress
        if epoch % save_interval == 0:
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            save_imgs(generator, epoch)




def save_imgs(generator, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, noise_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/angio_%d.png" % epoch)
    plt.close()
# Set your number of epochs and batch size here
# Set your number of epochs and batch size here
train(data, epochs=10000, batch_size=128, save_interval=100)

