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

class Unit3d(snt.AbstractModule):
    def  __init__(self,
                 output_channels ,
                 kernel_shape = (1,1,1),
                 stride = (1,1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 initializer = None,
                 use_bias = False,
                 name = 'unit_2d'):
        super(Unit3d, self).__init__(name=name)
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        self.initializer = initializer

    def _build(self,inputs,is_training):
        net = snt.Conv3D(output_channels=self.output_channels,
                         kernel_shape=self.kernel_shape,
                         stride=self.stride,
                         padding=snt.SAME,initializers=self.initializer,
                         use_bias=self.use_bias,regularizers=regularizers,name='conv_2d')(inputs)
        if self.use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net,is_training=is_training,test_local_stats=False)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        return net

class InceptionV1(snt.AbstractModule):

    VALID_ENDPOINTS = (
        'Conv2d_1a_7x7',
        'MaxPool2d_2a_3x3',
        'Conv2d_2b_1x1',
        'Conv2d_2c_3x3',
        'MaxPool2d_3a_3x3',
        'Mixed_2b',
        'Mixed_2c',
        'MaxPool2d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool2d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self,num_classes = 1000,spatia_squeeze = True,
                 final_endpoint = 'Logits',name = 'InceptionV1'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionV1, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint

    def _build(self, inputs ,is_training ,dropout_keep_prob = 1.0):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        net = inputs
        end_points = {}
        end_point = 'Conv2d_1a_7x7'
        net = Unit3d(output_channels=64,kernel_shape=[1,7,7],
                     stride=[1,2,2],name = end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        end_point = 'MaxPool2d_2a_3x3'
        net = tf.nn.max_pool3d(net,ksize=(1,1,3,3,1),strides=(1,1,2,2,1),
                               padding=snt.SAME,name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        end_point = 'Conv2d_2b_1x1'
        net =Unit3d(output_channels=64,kernel_shape=(1,1,1),
                    name=end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        end_point = 'Conv2d_2c_3x3'
        net = Unit3d(output_channels=192,kernel_shape=(1,3,3),
                     name=end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point:return net,end_points
        end_point = 'MaxPool2d_3a_3x3'
        net = tf.nn.max_pool3d(net,ksize=(1,1,3,3,1),strides=(1,1,2,2,1),
                               padding=snt.SAME,name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point:return net,end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3d(output_channels=64,kernel_shape=(1,1,1),
                                  name='Conv2d_0a_1x1')(net,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3d(output_channels=96,kernel_shape=(1,1,1),
                                  name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_1 = Unit3d(output_channels=128,kernel_shape=(1,3,3),
                                  name='Conv2d_0b_3x3')(branch_1,is_training=is_training)

            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3d(output_channels=16,kernel_shape=(1,1,1),
                                  name='Conv2d_0a_1x1')(net,is_training=is_training)
                branch_2 = Unit3d(output_channels=32,kernel_shape=(1,3,3),
                                  name='Conv2d_0b_3x3')(branch_2,is_training=is_training)

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net,ksize=(1,1,3,3,1),
                                            strides=(1,1,1,1,1),padding=snt.SAME,
                                            name='MaxPool2d_0a_3x3')
                branch_3 = Unit3d(output_channels=32,kernel_shape=(1,1,1),
                                  name='Conv2d_0b_1x1')(branch_3,is_training=is_training)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
            end_points[end_point] = net
            if self._final_endpoint == end_point:return net,end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3d(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3d(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3d(output_channels=192, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3d(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3d(output_channels=96, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool2d_0a_3x3')
                branch_3 = Unit3d(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool2d_4a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                               padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3d(output_channels=192, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3d(output_channels=96, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3d(output_channels=208, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3d(output_channels=16, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3d(output_channels=48, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool2d_0a_3x3')
                branch_3 = Unit3d(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3d(output_channels=160, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3d(output_channels=112, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3d(output_channels=224, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3d(output_channels=24, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3d(output_channels=64, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool2d_0a_3x3')
                branch_3 = Unit3d(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3d(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3d(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3d(output_channels=256, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3d(output_channels=24, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3d(output_channels=64, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool2d_0a_3x3')
                branch_3 = Unit3d(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3d(output_channels=112, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3d(output_channels=144, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3d(output_channels=288, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3d(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3d(output_channels=64, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool2d_0a_3x3')
                branch_3 = Unit3d(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3d(output_channels=256, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3d(output_channels=160, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3d(output_channels=320, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3d(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3d(output_channels=128, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool2d_0a_3x3')
                branch_3 = Unit3d(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool2d_5a_2x2'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1],
                               padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3d(output_channels=256, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3d(output_channels=160, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3d(output_channels=320, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3d(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3d(output_channels=128, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0a_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool2d_0a_3x3')
                branch_3 = Unit3d(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3d(output_channels=384, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3d(output_channels=192, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3d(output_channels=384, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3d(output_channels=48, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3d(output_channels=128, kernel_shape=[1, 3, 3],
                                  name='Conv2d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool2d_0a_3x3')
                branch_3 = Unit3d(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv2d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Logits'
        with tf.variable_scope(end_point):
            net = tf.nn.avg_pool3d(net,ksize=(1,1,7,7,1),
                                   strides=(1,1,1,1,1),padding=snt.VALID)
            net = tf.contrib.layers.dropout(net,keep_prob=dropout_keep_prob,is_training=is_training)
            logits = Unit3d(output_channels=self._num_classes,
                            kernel_shape=[1,1,1],
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='Conv2d_0c_1x1')(net,is_training=is_training)
            if self._spatia_squeeze:
                logits = tf.squeeze(logits,(2,3),name='SpatialSqueeze')
            averge_logits = tf.reduce_mean(logits,axis=1)
            end_points[end_point] = averge_logits
        if self._final_endpoint == end_point: return averge_logits, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax(averge_logits)
        end_points[end_point] = predictions
        return predictions,end_points





if __name__ == '__main__':
    # k = Unit3d(output_channels=3)
    import tensorflow as tf
    inputs = tf.placeholder(tf.float32,shape=(None,3,224,224,20))
    with tf.variable_scope('Flow'):
        inceptionv1 = InceptionV1(num_classes=1001)
        logits , e = inceptionv1(inputs,is_training=True)
    saver = tf.train.Saver(tf.global_variables(),reshape=True)
    for i in e:
        print(i,e[i])
    # with tf.Session() as sess:
    #     saver.restore(sess,r'E:\action_recognition\data\flow_inception_v1\flow_inception_v1.ckpt')
    # print(m(inputs,True))

    # print(InceptionV1.VALID_ENDPOINTS)
    #
    # for i in tf.global_variables():
    #     print(i )