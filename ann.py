import tensorflow as tf
import numpy as np

class ann:
    normal = lambda *s: tf.random_normal(s, stddev=0.09, dtype=tf.float32)
    normal_4bnn = lambda *s: tf.random_normal(s, stddev=1., dtype=tf.float32)
    bayes_prior_std = 100.
    
    def __init__(self, input_shape, output_shape, configuration, activation=None, 
                 bayesian=False, batchnorm=False, nsamples=2):
        if bayesian and batchnorm:
            raise(NotImplementedError("You shouldn't use batchnorm in bayesian networks"))
        
        configuration += [output_shape]
        
        self.weights = []
        self.offsnscales = []
        self.running_stats = []
        self.updates = []
        self.activation = activation
        self.loss = 0
        self.batchnorm = batchnorm
        self.bayesian = bayesian
        self.nsamples = nsamples
        
        normal = ann.normal
        normal_4bnn = ann.normal_4bnn
        
        def create_bn(inp_length):
            self.offsnscales.append([tf.Variable(normal(inp_length), name='mean_offset'), 
                                     tf.Variable(normal(inp_length) + 1, name='std_scale')])
            self.running_stats.append([tf.Variable(normal(inp_length), name='mean_sunning_stats'), 
                                       tf.Variable(normal(inp_length) + 1, name='std_sunning_stats')])

        with tf.variable_scope('layer_1'):
            if not bayesian:
                W, b = [tf.Variable(normal(input_shape, configuration[0]), name='W'), 
                        tf.Variable(normal(configuration[0]), name='b')]
            else:
                W, b, kl = self.construct_bayes_weights(input_shape, configuration[0], nsamples)
                self.loss += kl
            self.weights.append([W,b])
            if batchnorm:
                create_bn(configuration[1])

        for i, (inc, out) in enumerate(zip(configuration[:-1], configuration[1:])):
            with tf.variable_scope('layer_' + str(i+2)):
                if not bayesian:
                    W, b = tf.Variable(normal(inc, out), name='W'), tf.Variable(normal(out), name='b')
                else:
                    W, b, kl = self.construct_bayes_weights(inc, out, nsamples)
                    self.loss += kl
                self.weights.append([W,b])
                if batchnorm:
                    create_bn(out)

    def __call__(self, *x, mode=None):
        x = tf.concat(x, axis=-1)

        if self.bayesian:
            x = tf.stack([x]*self.nsamples, axis=0)

        def batch_matmul(x, y):
            temp = tf.transpose(tf.tensordot(x, y, axes=[[2], [1]]), perm=[0,2,1,3])
            ind = tf.stack([tf.range(tf.shape(temp)[0])]*2, axis=1)
            matrixes = tf.gather_nd(temp, indices=ind)
            return matrixes

        def apply_bn(x, offnscale, running_stats, mode):
            beta = 0.99

            if mode is None:
                raise(NotImplementedError('running mode mustbe provided to use batchnorm'))

            off, scale = offnscale
            
            tf.summary.histogram('offset', off)
            tf.summary.histogram('scale', scale)
            
            rmean, rvar = running_stats
            x_mean = tf.reduce_mean(x, axis=0)
            x_var = tf.reduce_mean((x - x_mean)**2, axis=0)

            mean = tf.where(mode, x_mean, rmean)
            var = tf.where(mode, x_var, rvar)

            x -= mean
            x /= tf.sqrt(var)
            self.updates += [rvar.assign(rvar*beta + x_var*(1-beta)), 
                             rmean.assign(rmean*beta + x_mean*(1-beta))]
            x = x*scale + off
            return x

        for i, (W, b) in enumerate(self.weights[:-1]):
            with tf.name_scope('layer_' + str(i+1)):
                if self.bayesian:
                    x = batch_matmul(x, W) + b[:,tf.newaxis,:]
                else:
                    x = tf.matmul(x, W) + b
                if self.batchnorm:
                    x = apply_bn(x, self.offsnscales[i], self.running_stats[i], mode)
                x = tf.nn.relu(x)

        W, b = self.weights[-1]
        with tf.name_scope('layer_' + str(len(self.weights))):
            if self.bayesian:
                x = batch_matmul(x, W) + b[:,tf.newaxis,:]
            else:
                x = tf.matmul(x, W) + b

            if self.batchnorm:
                x = apply_bn(x, self.offsnscales[-1], self.running_stats[-1], mode)
            self.logits = x

            if self.activation:
                return self.activation(x)

        return x
    
    @classmethod
    def construct_bayes_weights(cls, in_dim, out_dim, nsamples):
        kl = 0
        normal = cls.normal
        normal_4bnn = cls.normal_4bnn
        bayes_prior_std = cls.bayes_prior_std
        
        with tf.variable_scope('W'):
            Wmean = tf.Variable(normal(in_dim, out_dim))
            Wlogstd = tf.Variable(normal(in_dim, out_dim)-4)
            Wstd = tf.log1p(tf.exp(Wlogstd))

        kl += tf.reduce_sum(tf.log(bayes_prior_std/Wstd) + (Wstd**2 + Wmean**2)/(2*bayes_prior_std**2) - 0.5)
        
        with tf.variable_scope('b'):
            bmean = tf.Variable(normal(out_dim))
            blogstd = tf.Variable(normal(out_dim)-4)
            bstd = tf.log1p(tf.exp(blogstd))

        kl += tf.reduce_sum(tf.log(bayes_prior_std/bstd) + (bstd**2 + bmean**2)/(2*bayes_prior_std**2) - 0.5)
        
        
        W_distr = normal_4bnn(nsamples, in_dim, out_dim)*Wstd + Wmean
        b_distr = normal_4bnn(nsamples, out_dim)*bstd + bmean
        
        tf.summary.histogram('Wstd', Wstd)
        tf.summary.histogram('bstd', bstd)
        
        return W_distr, b_distr, kl