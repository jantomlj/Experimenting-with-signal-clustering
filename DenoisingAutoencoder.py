import tensorflow as tf
import numpy as np

def weight_variable(shape, name):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)

def vae_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim], layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        activations = act(preactivate, name='activation')
    return activations

class DenoisingAutoencoder:
    
    def __init__(self, encoder_layer1_size, encoder_layer2_size, encoder_layer3_size,
                 encoder_layer4_size, decoder_layer1_size, decoder_layer2_size, decoder_layer3_size, 
                 decoder_layer4_size, n_z, n_input, batch_size):
        self.expilon = 20.0
        self.n_z = n_z
        self.n_input = n_input
        self.batch_size = batch_size
        self.noise_sigma = 0.06
        
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        
        # Input
        self.x = tf.placeholder(tf.float32, shape=[None, n_input], name='input')
        
        #Gaussian noise
        #self.noise_sigma = 0.06
        #self.noise = tf.random_normal((self.batch_size, n_input), 0, self.noise_sigma, dtype=tf.float32)
        #self.input = tf.maximum(tf.minimum(self.x + self.noise, 1.0), 0.0)
        # NOTE SA OVIME, DOBIVA SE PRIRODNIH BAREM 7 CLUSTERA
        
        #Masking noise
        self.corruption = 0.25
        uni = tf.random_uniform((self.batch_size, n_input))
        bools = tf.greater(uni, self.corruption)
        self.noise = tf.cast(bools, tf.float32)
        self.input = self.x * self.noise
        
        
        # encoder
        self.layer_e1 = vae_layer(self.input, self.n_input, encoder_layer1_size, 'layer_e1') 
        self.layer_e2 = vae_layer(self.layer_e1, encoder_layer1_size, encoder_layer2_size, 'layer_e2')
        self.layer_e3 = vae_layer(self.layer_e2, encoder_layer2_size, encoder_layer3_size, 'layer_e3')
        self.layer_e4 = vae_layer(self.layer_e3, encoder_layer3_size, encoder_layer4_size, 'layer_e4')
        
        # latent state
        self.z = vae_layer(self.layer_e4, encoder_layer4_size, self.n_z, 'z', act=tf.identity)
        
        # decoder
        self.layer_d1 = vae_layer(self.z, self.n_z, decoder_layer1_size, 'layer_d1') 
        self.layer_d2 = vae_layer(self.layer_d1, decoder_layer1_size, decoder_layer2_size, 'layer_d2')
        self.layer_d3 = vae_layer(self.layer_d2, decoder_layer2_size, decoder_layer3_size, 'layer_d3')
        self.layer_d4 = vae_layer(self.layer_d3, decoder_layer3_size, decoder_layer4_size, 'layer_d4')
        
        # reconstruction
        self.x_reconstr = vae_layer(self.layer_d4, decoder_layer4_size, self.n_input, 'x_reconstr', act=tf.identity)
        
        # loss
        self.diff = (self.x - self.x_reconstr)**2
        
        self.cost1 = tf.reduce_sum(self.diff, 1)
        
        self.cost = tf.reduce_mean(self.cost1)  # average over batch
        
        
    def train(self, n_samples, data, n_epochs=10, learning_r=0.00005):
        local_data = np.copy(data)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_r).minimize(self.cost)

        init = tf.global_variables_initializer()                   
        self.sess.run(init)
        
        for epoch in range(n_epochs):
            avg_cost = 0.
            total_batch = n_samples // self.batch_size
            np.random.shuffle(local_data)

            print("Epoch " + str(epoch+1) + " started")
            
            data_pointer = 0
            for i in range(total_batch):
                batch_xs = local_data[data_pointer:(data_pointer + self.batch_size)]
                diff, opt, cos = self.sess.run((self.diff, self.optimizer, self.cost), feed_dict={self.x: batch_xs})
                avg_cost += cos / n_samples * self.batch_size
                
                data_pointer += self.batch_size
                
            print("Epoch: " + str(epoch + 1) + " cost = " + str(avg_cost)) 
            print("Average squared diff: " + str(np.average(diff)))
            
    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
    
    def load_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
    
    # data must be of batch_size length     
    def encodeBatchData(self, data):
        z = self.sess.run(self.z, feed_dict={self.x: data})
        return z
    
    # data must be of batch_size length
    def reconstructBatchData(self, data):
        x_reconstruct_mean = self.sess.run(self.x_reconstr, feed_dict={self.x: data})
        return x_reconstruct_mean

    # to encode all data, data must be of size K * batch_size, K integer
    def encode(self, data):
        result = np.array([]).reshape(0,self.n_z)
        pointer = 0
        total_batches = len(data) // self.batch_size
        for i in range(total_batches) :
            result = np.append(result, self.encodeBatchData(data[pointer:pointer+self.batch_size]), axis=0)
            pointer += self.batch_size
        return result
    
    # to encode all data, data must be of size K * batch_size, K integer
    def reconstruct(self, data):
        result = np.array([]).reshape(0,self.n_input)
        pointer = 0
        total_batches = len(data) // self.batch_size
        for i in range(total_batches) :
            result = np.append(result, self.reconstructBatchData(data[pointer:pointer+self.batch_size]), axis=0)
            pointer += self.batch_size
        return result
    