from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

initializer = {
    'w':tf.contrib.layers.xavier_initializer()
}
regularizers = {
    'w':tf.contrib.layers.l2_regularizer(scale=2e-4)
}

class Unit2d(snt.AbstractModule):
    def __init__(self,
                 output_channels ,
                 kernel_shape = (1,1),
                 stride = (1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 use_scale = False,
                 padding= snt.SAME,
                 name = 'unit_2d'):
        super(Unit2d, self).__init__(name=name)
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.padding = padding

    def _build(self,inputs,is_training):
        
        net = snt.Conv2D(output_channels=self.output_channels,
                            kernel_shape=self.kernel_shape,
                            stride=self.stride,
                            padding=self.padding,
                            use_bias=self.use_bias,regularizers=regularizers)(inputs)
        if self.use_batch_norm:
            bn = snt.BatchNorm(scale=self.use_scale)
            net = bn(net,is_training=is_training,test_local_stats=False)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        return net



class InceptionV3(snt.AbstractModule):

    VALID_ENDPOINTS = [
        'Conv2d_1a_3x3', 
        'Conv2d_2a_3x3', 
        'Conv2d_2b_3x3',
        'MaxPool_3a_3x3', 
        'Conv2d_3b_1x1', 
        'Conv2d_4a_3x3', 
        'MaxPool_5a_3x3',
        'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
        'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c',
        'Logits','Predictions']

    def __init__(self,num_classes = 101,spatia_squeeze = True,create_aux_logits = False,
                 final_endpoint = 'Logits',name = 'InceptionV3'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionV3, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint
        self._create_aux_logits = create_aux_logits

    def _build(self, inputs ,is_training ,dropout_keep_prob = 1.0):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        final_endpoint = self._final_endpoint
        
        net = inputs
        end_points = {}

        end_point = 'Conv2d_1a_3x3'
        net = Unit2d(32, [3, 3], stride=2,name=end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
        # 149 x 149 x 32
        end_point = 'Conv2d_2a_3x3'
        net = Unit2d(32, [3, 3], name=end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
        # 147 x 147 x 32
        end_point = 'Conv2d_2b_3x3'
        net = Unit2d(64, [3, 3], name=end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
        # 147 x 147 x 64
        end_point = 'MaxPool_3a_3x3'
        net = tf.nn.max_pool(net, [1,3, 3,1], strides=[1,2,2,1], padding=snt.SAME,name=end_point)
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
        # 73 x 73 x 64
        end_point = 'Conv2d_3b_1x1'
        net = Unit2d(80, [1, 1], name=end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
        # 73 x 73 x 80.
        end_point = 'Conv2d_4a_3x3'
        net = Unit2d(192, [3, 3],name=end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
        # 71 x 71 x 192.
        end_point = 'MaxPool_5a_3x3'
        net = tf.nn.max_pool(net, [1 ,3, 3, 1], strides=[1,2,2,1], padding=snt.SAME,name=end_point)
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
        # 35 x 35 x 192.

        # Inception blocks
        
        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(64, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(48, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_1 = Unit2d(64, [5, 5], name='Conv2d_0b_5x5')(branch_1,is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(64, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_2 = Unit2d(96, [3, 3], name='Conv2d_0b_3x3')(branch_2,is_training=is_training)
                branch_2 = Unit2d(96, [3, 3], name='Conv2d_0c_3x3')(branch_2,is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, [1,3, 3,1],[1,1,1,1], padding=snt.SAME,name='AvgPool_0a_3x3')
                branch_3 = Unit2d(32, [1, 1], name='Conv2d_0b_1x1')(branch_3,is_training=is_training)
            net = tf.concat(axis=-1, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        # mixed_1: 35 x 35 x 288.
        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(64, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(48, [1, 1], name='Conv2d_0b_1x1')(net,is_training=is_training)
                branch_1 = Unit2d(64, [5, 5], name='Conv_1_0c_5x5')(branch_1,is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(64, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_2 = Unit2d(96, [3, 3], name='Conv2d_0b_3x3')(branch_2,is_training=is_training)
                branch_2 = Unit2d(96, [3, 3], name='Conv2d_0c_3x3')(branch_2,is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, [1,3, 3,1],[1,1,1,1], padding=snt.SAME,name='AvgPool_0a_3x3')
                branch_3 = Unit2d(64, [1, 1], name='Conv2d_0b_1x1')(branch_3,is_training=is_training)
            net = tf.concat(axis=-1, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        # mixed_2: 35 x 35 x 288.
        end_point = 'Mixed_5d'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(64, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(48, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_1 = Unit2d(64, [5, 5], name='Conv2d_0b_5x5')(branch_1,is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(64, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_2 = Unit2d(96, [3, 3], name='Conv2d_0b_3x3')(branch_2,is_training=is_training)
                branch_2 = Unit2d(96, [3, 3], name='Conv2d_0c_3x3')(branch_2,is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, [1,3, 3,1],[1,1,1,1], padding=snt.SAME,name='AvgPool_0a_3x3')
                branch_3 = Unit2d(64, [1, 1], name='Conv2d_0b_1x1')(branch_3,is_training=is_training)
            net = tf.concat(axis=-1, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        # mixed_3: 17 x 17 x 768.
        end_point = 'Mixed_6a'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(384, [3, 3], stride=2, name='Conv2d_1a_1x1')(net,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(64, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_1 = Unit2d(96, [3, 3], name='Conv2d_0b_3x3')(branch_1,is_training=is_training)
                branch_1 = Unit2d(96, [3, 3], stride=2, name='Conv2d_1a_1x1')(branch_1,is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.nn.max_pool(net, [1,3, 3,1], strides=[1,2,2,1], padding=snt.SAME,name='MaxPool_1a_3x3')
            net = tf.concat(axis=-1, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        # mixed4: 17 x 17 x 768.
        end_point = 'Mixed_6b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(192, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(128, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_1 = Unit2d(128, [1, 7], name='Conv2d_0b_1x7')(branch_1,is_training=is_training)
                branch_1 = Unit2d(192, [7, 1], name='Conv2d_0c_7x1')(branch_1,is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(128, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_2 = Unit2d(128, [7, 1], name='Conv2d_0b_7x1')(branch_2,is_training=is_training)
                branch_2 = Unit2d(128, [1, 7], name='Conv2d_0c_1x7')(branch_2,is_training=is_training)
                branch_2 = Unit2d(128, [7, 1], name='Conv2d_0d_7x1')(branch_2,is_training=is_training)
                branch_2 = Unit2d(192, [1, 7], name='Conv2d_0e_1x7')(branch_2,is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, [1,3, 3,1],[1,1,1,1], padding=snt.SAME,name='AvgPool_0a_3x3')
                branch_3 = Unit2d(192, [1, 1],name='Conv2d_0b_1x1')(branch_3,is_training=is_training)
            net = tf.concat(axis=-1, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        # mixed_5: 17 x 17 x 768.
        end_point = 'Mixed_6c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(192, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(160, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_1 = Unit2d(160, [1, 7], name='Conv2d_0b_1x7')(branch_1,is_training=is_training)
                branch_1 = Unit2d(192, [7, 1], name='Conv2d_0c_7x1')(branch_1,is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(160, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_2 = Unit2d(160, [7, 1], name='Conv2d_0b_7x1')(branch_2,is_training=is_training)
                branch_2 = Unit2d(160, [1, 7], name='Conv2d_0c_1x7')(branch_2,is_training=is_training)
                branch_2 = Unit2d(160, [7, 1], name='Conv2d_0d_7x1')(branch_2,is_training=is_training)
                branch_2 = Unit2d(192, [1, 7], name='Conv2d_0e_1x7')(branch_2,is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, [1,3, 3,1],[1,1,1,1], padding=snt.SAME,name='AvgPool_0a_3x3')
                branch_3 = Unit2d(192, [1, 1],name='Conv2d_0b_1x1')(branch_3,is_training=is_training)
            net = tf.concat(axis=-1, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
        # mixed_6: 17 x 17 x 768.
        end_point = 'Mixed_6d'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(192, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(160, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_1 = Unit2d(160, [1, 7], name='Conv2d_0b_1x7')(branch_1,is_training=is_training)
                branch_1 = Unit2d(192, [7, 1], name='Conv2d_0c_7x1')(branch_1,is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(160, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_2 = Unit2d(160, [7, 1], name='Conv2d_0b_7x1')(branch_2,is_training=is_training)
                branch_2 = Unit2d(160, [1, 7], name='Conv2d_0c_1x7')(branch_2,is_training=is_training)
                branch_2 = Unit2d(160, [7, 1], name='Conv2d_0d_7x1')(branch_2,is_training=is_training)
                branch_2 = Unit2d(192, [1, 7], name='Conv2d_0e_1x7')(branch_2,is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, [1,3, 3,1],[1,1,1,1], padding=snt.SAME,name='AvgPool_0a_3x3')
                branch_3 = Unit2d(192, [1, 1],name='Conv2d_0b_1x1')(branch_3,is_training=is_training)
            net = tf.concat(axis=-1, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        # mixed_7: 17 x 17 x 768.
        end_point = 'Mixed_6e'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(192, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(192, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_1 = Unit2d(192, [1, 7], name='Conv2d_0b_1x7')(branch_1,is_training=is_training)
                branch_1 = Unit2d(192, [7, 1], name='Conv2d_0c_7x1')(branch_1,is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(192, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_2 = Unit2d(192, [7, 1], name='Conv2d_0b_7x1')(branch_2,is_training=is_training)
                branch_2 = Unit2d(192, [1, 7], name='Conv2d_0c_1x7')(branch_2,is_training=is_training)
                branch_2 = Unit2d(192, [7, 1], name='Conv2d_0d_7x1')(branch_2,is_training=is_training)
                branch_2 = Unit2d(192, [1, 7], name='Conv2d_0e_1x7')(branch_2,is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, [1,3, 3,1],[1,1,1,1], padding=snt.SAME,name='AvgPool_0a_3x3')
                branch_3 = Unit2d(192, [1, 1],name='Conv2d_0b_1x1')(branch_3,is_training=is_training)
            net = tf.concat(axis=-1, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        # mixed_8: 8 x 8 x 1280.
        end_point = 'Mixed_7a'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(192, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_0 = Unit2d(320, [3, 3], stride=2, name='Conv2d_1a_3x3')(branch_0,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(192, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_1 = Unit2d(192, [1, 7], name='Conv2d_0b_1x7')(branch_1,is_training=is_training)
                branch_1 = Unit2d(192, [7, 1], name='Conv2d_0c_7x1')(branch_1,is_training=is_training)
                branch_1 = Unit2d(192, [3, 3], stride=2, name='Conv2d_1a_3x3')(branch_1,is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.nn.max_pool(net, [1,3, 3,1], strides=[1,2,2,1], padding=snt.SAME,name='MaxPool_1a_3x3')
            net = tf.concat(axis=-1, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
        # mixed_9: 8 x 8 x 2048.
        end_point = 'Mixed_7b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(320, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(384, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_1 = tf.concat(axis=-1, values=[
                    Unit2d(384, [1, 3], name='Conv2d_0b_1x3')(branch_1,is_training=is_training),
                    Unit2d(384, [3, 1], name='Conv2d_0b_3x1')(branch_1,is_training=is_training)])
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(448, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_2 = Unit2d(384, [3, 3], name='Conv2d_0b_3x3')(branch_2,is_training=is_training)
                branch_2 = tf.concat(axis=-1, values=[
                    Unit2d(384, [1, 3], name='Conv2d_0c_1x3')(branch_2,is_training=is_training),
                    Unit2d(384, [3, 1], name='Conv2d_0d_3x1')(branch_2,is_training=is_training)])
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, [1,3, 3,1],[1,1,1,1], 'SAME',name='AvgPool_0a_3x3')
                branch_3 = Unit2d(192, [1, 1], name='Conv2d_0b_1x1')(branch_3,is_training=is_training)
            net = tf.concat(axis=-1, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        # mixed_10: 8 x 8 x 2048.
        end_point = 'Mixed_7c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(320, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(384, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_1 = tf.concat(axis=-1, values=[
                    Unit2d(384, [1, 3], name='Conv2d_0b_1x3')(branch_1,is_training=is_training),
                    Unit2d(384, [3, 1], name='Conv2d_0c_3x1')(branch_1,is_training=is_training)])
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(448, [1, 1], name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_2 = Unit2d(384, [3, 3], name='Conv2d_0b_3x3')(branch_2,is_training=is_training)
                branch_2 = tf.concat(axis=-1, values=[
                    Unit2d(384, [1, 3], name='Conv2d_0c_1x3')(branch_2,is_training=is_training),
                    Unit2d(384, [3, 1], name='Conv2d_0d_3x1')(branch_2,is_training=is_training)])
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, [1,3,3,1],[1,1,1,1], 'SAME',name='AvgPool_0a_3x3')
                branch_3 = Unit2d(192, [1, 1], name='Conv2d_0b_1x1')(branch_3,is_training=is_training)
            net = tf.concat(axis=-1, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
        
        if self._create_aux_logits: 
            with tf.variable_scope('AuxLogits'):
                aux = end_points['PreAuxLogits']
                aux = tf.nn.avg_pool(aux, [1,5,5,1], [1,3,3,1], padding='VALID', name='Conv2d_1a_3x3')
                aux = Unit2d(128, [1,1], name='Conv2d_1b_1x1')(aux,is_training=is_training)
                height , width = aux.shape.as_list()[1],aux.shape.as_list()[2]
                aux = Unit2d(768, [height,width], padding=snt.VALID,name='Conv2d_2a_5x5')(aux,is_training=is_training)
                aux = Unit2d(output_channels=self._num_classes,
                            kernel_shape=[1,1],
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='Logits')(aux,is_training=is_training)
            end_points['AuxLogits'] = aux

        end_point = 'Logits'
        with tf.variable_scope(end_point):
            net = tf.nn.avg_pool(net,ksize=(1,7,7,1),
                                   strides=(1,1,1,1),padding=snt.VALID)
            net = tf.contrib.layers.dropout(net,keep_prob=dropout_keep_prob,is_training=is_training)
            logits = Unit2d(output_channels=self._num_classes,
                            kernel_shape=[1,1],
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='Conv2d_1c_1x1')(net,is_training=is_training)
            if self._spatia_squeeze:
                logits = tf.squeeze(logits,(1,2),name='SpatialSqueeze')
            averge_logits = logits
            end_points[end_point] = averge_logits
        if self._final_endpoint == end_point: return averge_logits, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax(averge_logits)
        end_points[end_point] = predictions
        return predictions,end_points

if __name__ == "__main__":
    inputs = tf.placeholder(tf.float32,shape=(None,224,224,10))
    with tf.variable_scope('Flow'):
        _ ,e = InceptionV3(num_classes=1001)(inputs,is_training=True)
    
    g = sorted(e)
    for i in g:
        print(i,e[i])
    
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # rgb_saver.restore(sess,'/mnt/zhujian/ckpt/flow_snt_inception_v3/model.ckpt')
    # print(sess.run(tf.losses.get_regularization_loss()))