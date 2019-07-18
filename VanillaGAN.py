import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import scipy.misc
import os

result_path = './result/vanilla-gan/'

# fixed random seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# hyperparameter
input_shape = [-1, 28, 28, 1]
d_learning_rate = 1e-4
g_learning_rate = 1e-4
batch_size = 64
iteration = 100000
z_dim = 100

# gan models
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(1)
        self.dropout = layers.Dropout(0.2)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = tf.nn.sigmoid(x)
        
        return x


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(784)
        self.dropout = layers.Dropout(0.2)
    
    def call(self, x, training=False):
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = tf.nn.sigmoid(x)
        
        return x


def sampling_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

    

G = Generator()
G.build(input_shape=(batch_size, z_dim))
D = Discriminator()
D.build(input_shape=(batch_size, 28, 28, 1))

g_optimizer = tf.optimizers.Adam(g_learning_rate)
d_optimizer = tf.optimizers.Adam(d_learning_rate)


# load dataset
(x_train, _), (x_test ) = keras.datasets.mnist.load_data()
x_train = tf.cast(x_train, np.float32) / 255.
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(batch_size * 32).repeat().batch(batch_size)
dataset = tf.compat.v1.data.make_one_shot_iterator(dataset)


d_losses = []
g_losses = []
# training
for itr in range(iteration):
    images = dataset.get_next()

    real_batch = tf.reshape(images, shape=input_shape)

    # training Discriminator
    with tf.GradientTape() as tape:
        z = sampling_z(batch_size, z_dim)
        fake_batch = G(z)
        fake_batch = tf.reshape(fake_batch, input_shape)
        D_real_logits = D(real_batch, training=True)
        D_fake_logits = D(fake_batch, training=True)
        D_loss = -tf.reduce_mean(tf.math.log(D_real_logits) + tf.math.log(1. - D_fake_logits))

    grads = tape.gradient(D_loss, D.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, D.trainable_variables))
    d_losses.append(D_loss)

    # training Generator
    with tf.GradientTape() as tape:
        z = sampling_z(batch_size, z_dim)
        fake_batch = G(z, training=True)
        fake_batch = tf.reshape(fake_batch, input_shape)
        D_fake_logits = D(fake_batch)
        G_loss = -tf.reduce_mean(tf.math.log(D_fake_logits))

    grads = tape.gradient(G_loss, G.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, G.trainable_variables))
    g_losses.append(G_loss)


    if itr % 100 == 0:
        z = sampling_z(1, z_dim)
        image = G(z)
        image = tf.reshape(image, input_shape);
        image = tf.squeeze(image)
        scipy.misc.imsave(os.path.join(result_path, '{}.jpg'.format(itr)), image)
        print("Iter: {}    D_loss: {:.4f}    G_loss: {:.4f}".format(itr, D_loss, G_loss))


        
plt.plot(d_losses)
plt.plot(g_losses)
plt.title('GAN Loss')

plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.legend(['Discriminator', 'Generator'], loc='upper right')
plt.show()


    

