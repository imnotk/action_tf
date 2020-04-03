from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sonnet as snt
import tensorflow as tf
import numpy as np

class i2d(snt.AbstractModule):
    def __init__(self,
                 output_channels ,
                 kernel_size = (1,1),
                 stride = (1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 name = 'unit_2d'):
        super(i2d, self).__init__(name=name)
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias


    def _build(self,inputs,is_training):
        net = snt.Conv2D(output_channels=self.output_channels,
                         kernel_shape=self.kernel_size,
                         stride=self.stride,
                         padding=snt.SAME,
                         use_bias=self.use_bias)(inputs)
        if self.use_batch_norm:
            bn = snt.BatchNorm()
            # net = tf.layers.batch_normalization(inputs,training=is_training)
            net = bn(net,is_training=is_training,test_local_stats=True)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        return net

class vgg16(snt.AbstractModule):

    VALID_ENDPOINT ={
        'Conv_1_2',
        'MaxPool_1',
        'Conv_2_2',
        'MaxPool_2',
        'Conv_3_3',
        'MaxPool_3',
        'Conv_4_3',
        'MaxPool_4',
        'Conv_5_3',
        'MaxPool_5',
        'FC',
        'Predictions'
    }

    def __init__(self,num_classes = 101,
                 final_endpoint = 'FC',name = 'vgg16'):
        if final_endpoint not in self.VALID_ENDPOINT:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(vgg16, self).__init__(name=name)
        self.num_classes = num_classes
        self.final_endpoint = final_endpoint

    def _build(self, inputs,is_training,dropout_keep_prob = 0.5):
        if self.final_endpoint not in self.VALID_ENDPOINT:
            raise  ValueError('Unknown final endpoint %s' % self.final_endpoint)

        net = inputs
        end_points = {}
        end_point = 'Conv_1_2'
        for _ in range(2):
            net = i2d(output_channels=64,kernel_size=(3,3),stride=(1,1),
                      name = end_point)(net,is_training=is_training)
        #  iccv_2017 two-stream flow-guided attention networks
        attention_net = net
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net,end_points
        end_point = 'MaxPool_1'
        net = tf.nn.max_pool(net,ksize=[1,2,2,1],
                             strides=[1,2,2,1],padding=snt.SAME,name=end_point)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net,end_points
        end_point = 'Conv_2_2'
        for _ in range(2):
            net = i2d(output_channels=128,kernel_size=(3,3),stride=(1,1),
                      name = end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net,end_points
        end_point = 'MaxPool_2'
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points
        end_point = 'Conv_3_3'
        for _ in range(2):
            net = i2d(output_channels=256, kernel_size=(3, 3), stride=(1, 1),
                      name=end_point)(net, is_training=is_training)
        net = i2d(output_channels=256,kernel_size=(1,1),stride=(1,1),
                  name=end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_3'
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points
        end_point = 'Conv_4_3'
        for _ in range(2):
            net = i2d(output_channels=512, kernel_size=(3, 3), stride=(1, 1),
                      name=end_point)(net, is_training=is_training)
        net = i2d(output_channels=512, kernel_size=(1, 1), stride=(1, 1),
                  name=end_point)(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_4'
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points
        end_point = 'Conv_5_3'
        for _ in range(2):
            net = i2d(output_channels=512, kernel_size=(3, 3), stride=(1, 1),
                      name=end_point)(net, is_training=is_training)
        net = i2d(output_channels=512, kernel_size=(1, 1), stride=(1, 1),
                  name=end_point)(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_5'
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points
        end_point = 'FC'
        shp = net.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        net = tf.reshape(net,[-1,flattened_shape],name=end_point)
        net = tf.layers.Dense(units=4096,activation=tf.nn.relu,name=end_point)(net)
        net = tf.nn.dropout(net,dropout_keep_prob)
        net = tf.layers.Dense(units=4096,activation=tf.nn.relu,name=end_point)(net)
        net = tf.nn.dropout(net,dropout_keep_prob)
        net = tf.layers.Dense(units=self.num_classes,name=end_point)(net)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points
        end_point = 'Predictions'
        predicitons = tf.nn.softmax(net)
        end_points[end_point] = predicitons
        return net,end_points



if __name__ == '__main__':
    i2d_model = vgg16(
        num_classes=101,final_endpoint='Predictions'
    )
    inp = tf.placeholder(tf.float32,[None,224,224,3])
    predictions,end_points = i2d_model(
        inp,is_training = True,dropout_keep_prob = 1.0
    )
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sample_input = np.zeros((5,224,224,3))
        out_predictions , out_logits = sess.run(
            [predictions,end_points['FC']],{inp:sample_input}
        )

        print(out_predictions)
        print('what`s next')
        print(out_logits.shape)