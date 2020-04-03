from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

initializer = {
    'w':tf.contrib.layers.xavier_initializer()
}
regularizers = {
    'w':tf.contrib.layers.l2_regularizer(scale=5e-4)
}

class Unit2d(snt.AbstractModule):
    def __init__(self,
                 output_channels ,
                 kernel_shape = (1,1),
                 stride = (1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 depth_multiplier = 1,
                 use_scale = False,
                 name = 'unit_2d',
                 bn_trainable=True,
                 separable = False):
        super(Unit2d, self).__init__(name=name)
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        self.depthwise_multiplier = depth_multiplier
        self.use_pbn = bn_trainable
        self.use_scale = use_scale
        self._separable = separable

    def _build(self,inputs,is_training):
        if self._separable:
            net = snt.SeparableConv2D(output_channels=self.output_channels,
                                      channel_multiplier=8,
                                      kernel_shape=self.kernel_shape,
                                      stride=self.stride,padding=snt.SAME,
                                      use_bias=self.use_bias)(inputs)
        else:
            net = snt.Conv2D(output_channels=self.output_channels,
                                kernel_shape=self.kernel_shape,
                                stride=self.stride,
                                padding=snt.SAME,
                                use_bias=self.use_bias,regularizers=regularizers)(inputs)
        if self.use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net,is_training=is_training,test_local_stats=False)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        return net

class InceptionV2(snt.AbstractModule):

    VALID_ENDPOINTS = (
        'Conv2d_1a_7x7',
        'MaxPool_2a_3x3',
        'Conv2d_2b_1x1',
        'Conv2d_2c_3x3',
        'MaxPool_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'Mixed_4a',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_5a',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self,num_classes = 101,spatia_squeeze = True,use_pbn=False,
                 final_endpoint = 'Logits',name = 'InceptionV2'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionV2, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint
        self.use_pbn = use_pbn

    def _build(self, inputs ,is_training ,dropout_keep_prob = 1.0):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        if self.use_pbn:
            _use_bn = False
        else:
            _use_bn = True

        net = inputs
        end_points = {}
        end_point = 'Conv2d_1a_7x7'
        net = Unit2d(output_channels=64,kernel_shape=[7,7],use_batch_norm=False,separable=True,use_bias=True,
                     stride=[2,2],name = end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points 
        end_point = 'MaxPool_2a_3x3'
        net = tf.nn.max_pool(net,ksize=(1,3,3,1),strides=(1,2,2,1),
                               padding=snt.SAME,name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        end_point = 'Conv2d_2b_1x1'
        net =Unit2d(output_channels=64,kernel_shape=(1,1),
                    name=end_point)(net,is_training=is_training & _use_bn)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        end_point = 'Conv2d_2c_3x3'
        net = Unit2d(output_channels=192,kernel_shape=(3,3),
                     name=end_point)(net,is_training=is_training & _use_bn)
        end_points[end_point] = net
        if self._final_endpoint == end_point:return net,end_points
        end_point = 'MaxPool_3a_3x3'
        net = tf.nn.max_pool(net,ksize=(1,3,3,1),strides=(1,2,2,1),
                               padding=snt.SAME,name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point:return net,end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(output_channels=64,kernel_shape=(1,1),
                                      name='Conv2d_0a_1x1')(net,is_training=is_training & _use_bn )
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(output_channels=64,kernel_shape=(1,1),
                                  name='Conv2d_0a_1x1')(net,is_training=is_training & _use_bn )
                branch_1 = Unit2d(output_channels=64,kernel_shape=(3,3),
                                  name='Conv2d_0b_3x3')(branch_1,is_training=is_training & _use_bn )

            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(output_channels=64,kernel_shape=(1,1),
                                  name='Conv2d_0a_1x1')(net,is_training=is_training & _use_bn )
                branch_2 = Unit2d(output_channels=96,kernel_shape=(3,3),
                                  name='Conv2d_0b_3x3')(branch_2,is_training=is_training & _use_bn )
                branch_2 = Unit2d (output_channels=96, kernel_shape=(3, 3),
                                   name='Conv2d_0c_3x3') (branch_2, is_training=is_training & _use_bn )

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net,ksize=(1,3,3,1),
                                            strides=(1,1,1,1),padding=snt.SAME,
                                            name='AvgPool_0a_3x3')
                branch_3 = Unit2d(output_channels=32,kernel_shape=(1,1),
                                  name='Conv2d_0b_1x1')(branch_3,is_training=is_training & _use_bn )

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
            end_points[end_point] = net
            if self._final_endpoint == end_point:return net,end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(output_channels=64, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn )
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(output_channels=64, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn )
                branch_1 = Unit2d(output_channels=96, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training & _use_bn )
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(output_channels=64, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn )
                branch_2 = Unit2d(output_channels=96, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training & _use_bn )
                branch_2 = Unit2d (output_channels=96, kernel_shape=[3, 3],
                                   name='Conv2d_0c_3x3') (branch_2,
                                                          is_training=is_training & _use_bn )
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1],
                                            strides=[1, 1, 1, 1], padding=snt.SAME,
                                            name='AvgPool_0a_3x3')
                branch_3 = Unit2d(output_channels=64, kernel_shape=[1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training & _use_bn )
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4a'
        with tf.variable_scope (end_point):
            with tf.variable_scope ('Branch_0'):
                branch_0 = Unit2d (output_channels=128, kernel_shape=[1, 1],
                                   name='Conv2d_0a_1x1') (net, is_training=is_training & _use_bn )
                branch_0 = Unit2d (output_channels=160, kernel_shape=[3, 3],stride=[2,2],
                                   name='Conv2d_1a_3x3') (branch_0, is_training=is_training & _use_bn )

            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(output_channels=64, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn )
                branch_1 = Unit2d(output_channels=96, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training & _use_bn)
                branch_1 = Unit2d (output_channels=96, kernel_shape=[3, 3],stride=[2,2],
                                   name='Conv2d_1a_3x3') (branch_1,
                                                          is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.nn.max_pool(net, ksize=[1, 3, 3, 1],
                                            strides=[1, 2, 2, 1], padding=snt.SAME,
                                            name='MaxPool_0a_3x3')

            net = tf.concat ([branch_0, branch_1, branch_2], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(output_channels=224, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(output_channels=64, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
                branch_1 = Unit2d(output_channels=96, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(output_channels=96, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
                branch_2 = Unit2d(output_channels=128, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training & _use_bn)
                branch_2 = Unit2d (output_channels=128, kernel_shape=[3, 3],
                                   name='Conv2d_0c_3x3') (branch_2,
                                                          is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1],
                                            strides=[1, 1, 1, 1], padding=snt.SAME,
                                            name='AvgPool_0a_3x3')
                branch_3 = Unit2d(output_channels=128, kernel_shape=[1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training & _use_bn)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(output_channels=192, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(output_channels=96, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
                branch_1 = Unit2d(output_channels=128, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(output_channels=96, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
                branch_2 = Unit2d(output_channels=128, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training & _use_bn)
                branch_2 = Unit2d (output_channels=128, kernel_shape=[3, 3],
                                   name='Conv2d_0c_3x3') (branch_2,
                                                          is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1],
                                            strides=[1, 1, 1, 1], padding=snt.SAME,
                                            name='AvgPool_0a_3x3')
                branch_3 = Unit2d(output_channels=128, kernel_shape=[1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training & _use_bn)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(output_channels=160, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(output_channels=128, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
                branch_1 = Unit2d(output_channels=160, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(output_channels=128, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
                branch_2 = Unit2d(output_channels=160, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training & _use_bn)
                branch_2 = Unit2d (output_channels=160, kernel_shape=[3, 3],
                                   name='Conv2d_0c_3x3') (branch_2,
                                                          is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1],
                                            strides=[1, 1, 1, 1], padding=snt.SAME,
                                            name='AvgPool_0a_3x3')
                branch_3 = Unit2d(output_channels=96, kernel_shape=[1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training & _use_bn)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(output_channels=96, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(output_channels=128, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
                branch_1 = Unit2d(output_channels=192, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(output_channels=160, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
                branch_2 = Unit2d(output_channels=192, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training & _use_bn)
                branch_2 = Unit2d (output_channels=192, kernel_shape=[3, 3],
                                   name='Conv2d_0c_3x3') (branch_2,
                                                          is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1],
                                            strides=[1, 1, 1, 1], padding=snt.SAME,
                                            name='AvgPool_0a_3x3')
                branch_3 = Unit2d(output_channels=96, kernel_shape=[1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training & _use_bn)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5a'
        with tf.variable_scope (end_point):
            with tf.variable_scope ('Branch_0'):
                branch_0 = Unit2d (output_channels=128, kernel_shape=[1, 1],
                                   name='Conv2d_0a_1x1') (net, is_training=is_training & _use_bn)
                branch_0 = Unit2d (output_channels=192, kernel_shape=[3, 3],stride=[2,2],
                                   name='Conv2d_1a_3x3') (branch_0, is_training=is_training & _use_bn)
            with tf.variable_scope ('Branch_1'):
                branch_1 = Unit2d (output_channels=192, kernel_shape=[1, 1],
                                   name='Conv2d_0a_1x1') (net, is_training=is_training & _use_bn)
                branch_1 = Unit2d (output_channels=256, kernel_shape=[3, 3],
                                   name='Conv2d_0b_3x3') (branch_1,
                                                          is_training=is_training & _use_bn)
                branch_1 = Unit2d (output_channels=256, kernel_shape=[3, 3],stride=[2,2],
                                   name='Conv2d_1a_3x3') (branch_1,
                                                          is_training=is_training & _use_bn)

            with tf.variable_scope ('Branch_2'):
                branch_2 = tf.nn.max_pool (net, ksize=[1, 3, 3, 1],
                                           strides=[1, 2, 2, 1], padding=snt.SAME,
                                           name='MaxPool_0a_3x3')

            net = tf.concat ([branch_0, branch_1, branch_2], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(output_channels=352, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(output_channels=192, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
                branch_1 = Unit2d(output_channels=320, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(output_channels=160, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
                branch_2 = Unit2d(output_channels=224, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training & _use_bn)
                branch_2 = Unit2d (output_channels=224, kernel_shape=[3, 3],
                                   name='Conv2d_0c_3x3') (branch_2,
                                                          is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1],
                                            strides=[1, 1, 1, 1], padding=snt.SAME,
                                            name='AvgPool_0a_3x3')
                branch_3 = Unit2d(output_channels=128, kernel_shape=[1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training & _use_bn)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit2d(output_channels=352, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit2d(output_channels=192, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
                branch_1 = Unit2d(output_channels=320, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit2d(output_channels=192, kernel_shape=[1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training & _use_bn)
                branch_2 = Unit2d(output_channels=224, kernel_shape=[3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training & _use_bn)
                branch_2 = Unit2d (output_channels=224, kernel_shape=[3, 3],
                                   name='Conv2d_0c_3x3') (branch_2,
                                                          is_training=is_training & _use_bn)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool(net, ksize=[1, 3, 3, 1],
                                            strides=[1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool_0a_3x3')
                branch_3 = Unit2d(output_channels=128, kernel_shape=[1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training & _use_bn)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Logits'
        with tf.variable_scope(end_point):
            net = tf.nn.avg_pool(net,ksize=(1,7,7,1),
                                   strides=(1,1,1,1),padding=snt.VALID)
            # net = tf.nn.dropout(net,dropout_keep_prob)
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





if __name__ == '__main__':
    import tensorflow as tf
    inputs = tf.placeholder(tf.float32,shape=(None,224,224,10))
    with tf.variable_scope('Flow'):
        _ ,e = InceptionV2(num_classes=1001)(inputs,is_training=True)
        rgb_var_map = {}
        for var in tf.global_variables ():
            if var.name.split ('/')[0] == 'Flow':
                rgb_var_map[var.name.replace (':0', '')] = var
        rgb_saver = tf.train.Saver (var_list=rgb_var_map, reshape=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(r'/mnt/zhujian/ckpt/flow_snt_inception_v2')
        if ckpt is not None:
            rgb_saver.restore(sess,ckpt.model_checkpoint_path)
    for i in tf.global_variables():
        print(i)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    r_loss = sess.run(tf.losses.get_regularization_loss ())
    print(r_loss)