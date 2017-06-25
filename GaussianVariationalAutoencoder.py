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

class GaussianVariationalAutoencoder:
    
    def __init__(self, encoder_layer1_size, encoder_layer2_size, encoder_layer3_size,
                 encoder_layer4_size, decoder_layer1_size, decoder_layer2_size, decoder_layer3_size, 
                 decoder_layer4_size, n_z, n_input, batch_size):
        self.expilon = 16.0
        self.n_z = n_z
        self.n_input = n_input
        self.batch_size = batch_size
        
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        
        # Input
        self.x = tf.placeholder(tf.float32, shape=[None, n_input], name='input')
        
        # encoder
        self.layer_e1 = vae_layer(self.x, self.n_input, encoder_layer1_size, 'layer_e1') 
        self.layer_e2 = vae_layer(self.layer_e1, encoder_layer1_size, encoder_layer2_size, 'layer_e2')
        self.layer_e3 = vae_layer(self.layer_e2, encoder_layer2_size, encoder_layer3_size, 'layer_e3')
        self.layer_e4 = vae_layer(self.layer_e3, encoder_layer3_size, encoder_layer4_size, 'layer_e4')
        
        # latent state
        self.z_mean = vae_layer(self.layer_e4, encoder_layer4_size, self.n_z, 'z_mean', act=tf.identity)
        self.z_log_sigma_sq = vae_layer(self.layer_e4, encoder_layer4_size, self.n_z, 'z_log_sigma_sq', act=tf.identity)
        self.eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(self.eps, self.z_log_sigma_sq))
        
        # decoder
        self.layer_d1 = vae_layer(self.z, self.n_z, decoder_layer1_size, 'layer_d1') 
        self.layer_d2 = vae_layer(self.layer_d1, decoder_layer1_size, decoder_layer2_size, 'layer_d2')
        self.layer_d3 = vae_layer(self.layer_d2, decoder_layer2_size, decoder_layer3_size, 'layer_d3')
        self.layer_d4 = vae_layer(self.layer_d3, decoder_layer3_size, decoder_layer4_size, 'layer_d4')
        
        # reconstruction
        self.x_reconstr_mean = vae_layer(self.layer_d4, decoder_layer4_size, self.n_input, 'x_reconstr_mean', act=tf.identity)
        self.x_reconstr_log_sigma_sq = vae_layer(self.layer_d4, decoder_layer4_size, self.n_input, 'x_reconstr_log_sigma_sq', 
                                            act=tf.identity)
        
        # loss
        self.diff = (self.x - self.x_reconstr_mean)**2
        self.partCost = tf.exp(tf.minimum(self.x_reconstr_log_sigma_sq, self.expilon))
        self.cost1 = tf.reduce_sum(0.5 * self.x_reconstr_log_sigma_sq + (0.5 * 
                                        (self.diff / 
                                        self.partCost)), 1)
        
        #self.constrained_x_reconstr_log_sigma_sq = tf.minimum(self.x_reconstr_log_sigma_sq, self.expilon)
        #self.x_sigma = tf.sqrt(tf.exp(self.constrained_x_reconstr_log_sigma_sq))
        
        #self.diff = (self.x - self.x_reconstr_mean)**2
        #self.nominator = tf.exp(tf.minimum((self.diff) / (self.x_sigma + 1e-8), self.expilon))
        #self.denominator = self.x_sigma * 2.5066283
        #self.division = self.nominator / (self.denominator)
        #self.logarithm = -tf.log(self.division)
        
        #self.cost1 = tf.reduce_sum(self.diff, 1) #- tf.reduce_sum(self.logarithm, 1)
        
        self.cost2 = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - 
                                          tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(self.cost1 + self.cost2)  # average over batch
        
        
    def bigger(self, x):
        return x<-1e5 or x>1e5
    def bigger2(self, x):
        return x<0.001 and x>-0.001
    def bigger3(self, x):
        return x<-1e1 or x>1e1
    
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
                #debug1, debug2, nom, mygrad, sigma, opt, cos = self.sess.run((self.diff, self.layer_e1, self.nominator, my_gradient, self.x_sigma, self.optimizer, self.cost), feed_dict={self.x: batch_xs})
                diff, opt, cos = self.sess.run((self.diff, self.optimizer, self.cost), feed_dict={self.x: batch_xs})
                avg_cost += cos / n_samples * self.batch_size
                #print("i: " + str(i))
                #print("Debug1: " + str(debug1) + ",\n debug2: " + str(debug2))
                #print("My gradient: " + str(mygrad))
                #fun = np.vectorize(self.bigger)
                #fun2 = np.vectorize(self.bigger)
                #fun3 = np.vectorize(self.bigger3)
                #condition = fun(mygrad)
                #condition2 = fun2(nom)
                #condition3 = fun3(debug1)
                #print("Sigma: " + str(sigma))
                #print("Debug1, big/small: " + str(np.extract(condition3, debug1)))
                #print("My gradients, big/small: " + str(np.extract(condition, mygrad)))
                #print("Nominator: " + str(nom))
                #print("Nominator, big/small: " + str(np.extract(condition2, nom)))
                #print("does My gradient contain Nan: " + str(np.isnan(mygrad).any()))
                #print("does Debug1 contain Nan: " + str(np.isnan(debug1).any()))
                
                
                data_pointer += self.batch_size
                #if np.isnan(debug2).any():
                #    exit()
                
            print("Epoch: " + str(epoch + 1) + " cost = " + str(avg_cost)) 
            print("Average squared diff: " + str(np.average(diff)))
    # data must be of batch_size length     
    def encodeBatchData(self, data):
        z = self.sess.run(self.z, feed_dict={self.x: data})
        return z
    
    # data must be of batch_size length
    def reconstructBatchData(self, data):
        x_reconstruct_mean = self.sess.run(self.x_reconstr_mean, feed_dict={self.x: data})
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