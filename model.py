import tensorflow as tf

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(dtype=tf.float32, shape=(None,real_dim), name='input_real')
    inputs_z = tf.placeholder(dtype=tf.float32, shape=(None,z_dim),name='input_z')
    return inputs_real, inputs_z

def generator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):       # Putting variables in same scope
        h1 = tf.layers.dense(z,n_units,activation=None)     # Hidden Layer
        h1 = tf.maximum(alpha*h1,h1)        # Leaky ReLU
        logits = tf.layers.dense(h1,out_dim,activation=None)    # Logits
        output = tf.tanh(logits)        # Output layer ranges from [-1,1]
        return output

def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):   # Putting variables in same scope
        h1 = tf.layers.dense(x,n_units,activation=None)     # Hidden Layer
        h1 = tf.maximum(alpha*h1,h1)        # Leaky ReLU
        logits = tf.layers.dense(h1,1,activation=None)      # Logits
        output = tf.sigmoid(logits)         # Output varies from [0,1]
        return output, logits
    
