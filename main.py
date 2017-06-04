import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data      # Importing MNIST Datasets
mnist = input_data.read_data_sets('MNIST_data')

from model import *     # Importing our model



""" Hyperparameters """

# Size of input image to discriminator
input_size = 784        # 28x28 MNIST images flattened

# Size of latent vector to generator
z_size = 100

# Sizes of hidden layers in generator and discriminator
g_hidden_size = 128
d_hidden_size = 128

# Leak factor for leaky ReLU
alpha = 0.01

# Label smoothing 
smooth = 0.1
# it is used so as to create a probablity distribution easily rather than having a absolute certainity about any real or fake case



""" Building Network """

tf.reset_default_graph()

# Create our input placeholders
input_real, input_z = model_inputs(input_size, z_size)

# Generator network
g_model = geneartor(input_z, input_size)        #g_model is Generator output

# Discriminator network
d_model_real, d_logits_real = discriminator(input_real)
d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)



""" Discriminator and Generator Losses """

# discriminator loss = d_loss_real + d_loss_fake

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_logits_real)*(1-smooth)))

d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_logits_fake)))

d_loss = d_loss_real + d_loss_fake

# generator loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, labels=tf.ones_like(d_logits_fake)))



""" Optimizers """

# Learning Rate
learning_rate = 0.002

# Getting Trainable Variables
# All Variables
t_vars = tf.trainable_variables()

# All variables with 'generator' scope
g_vars = [var for var in t_vars if var.name.startswith('generator')]

# All variables with 'discriminator' scope
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]


# Optimizing loss in dicriminator and generator seperately

# Optimizing Disciminator loss
d_train_opt = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)

# Optimizing Generator loss 
g_train_opt = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars) 



""" Training """

batch_size = 100
epochs = 100
samples = []
losses = []
saver = tf.train.Saver(var_list = g_vars)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)
            
            # Get images, reshape and rescale to pass to D
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images*2 - 1
            
            # Sample random noise for G
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            
            # Run optimizers
            _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
            _ = sess.run(g_train_opt, feed_dict={input_z: batch_z})
        
        # At the end of each epoch, get the losses and print them out
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})
            
        print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))    
        # Save losses to view after training
        losses.append((train_loss_d, train_loss_g))
        
        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(
                       generator(input_z, input_size, reuse=True),
                       feed_dict={input_z: sample_z})
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)



""" Training Loss """
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()



""" Generator samples from training """

# View of Samples
def view_samples(epoch, samples):
    fig, axes = plt.subplot(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax,img in zip(flatten(), samples[epoch]):
        ax.xaxis.set_visiblity(False)
        ax.yaxis.set_visiblity(False)
        im = ax.imshow(img.reshape(28,28), cmap='Grey_r')
    
    return fig, axes

# Load samples from generator taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

_ = view_samples(-1, samples)

# Generated Images as network was training
rows, cols = 10, 6
fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)



""" Sampling from Generator """

saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    sample_z = np.random.uniform(-1, 1, size=(16, z_size))
    gen_samples = sess.run(
                   generator(input_z, input_size, reuse=True),
                   feed_dict={input_z: sample_z})
view_samples(0, [gen_samples])