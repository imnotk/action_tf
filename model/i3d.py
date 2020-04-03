from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

regularizers = {
    'w':tf.contrib.layers.l2_regularizer(scale=5e-4)
}

class Unit3D(snt.AbstractModule):

    def __init__(self,output_channels,
                 kernel_shape = (1,1,1),
                 stride = (1,1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 name = 'unit_3d'):
        super(Unit3D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias

    def _build(self, inputs , is_training):
        net = snt.Conv3D(output_channels=self._output_channels,
                         kernel_shape=self._kernel_shape,
                         stride=self._stride,
                         padding=snt.SAME,regularizers=regularizers,
                         use_bias=self._use_bias)(inputs)
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net,is_training = is_training,test_local_stats = False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net

class InceptionI3d(snt.AbstractModule):

    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self,num_classes = 400,spatia_squeeze = True,
                 final_endpoint = 'Logits',name = 'inception_i3d'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint

    def _build(self, inputs ,is_training ,dropout_keep_prob = 1.0):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        net = inputs
        end_points = {}
        end_point = 'Conv3d_1a_7x7'
        net = Unit3D(output_channels=64,kernel_shape=[7,7,7],
                     stride=[2,2,2],name = end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        end_point = 'MaxPool3d_2a_3x3'
        net = tf.nn.max_pool3d(net,ksize=(1,1,3,3,1),strides=(1,1,2,2,1),
                               padding=snt.SAME,name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        end_point = 'Conv3d_2b_1x1'
        net =Unit3D(output_channels=64,kernel_shape=(1,1,1),
                    name=end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        end_point = 'Conv3d_2c_3x3'
        net = Unit3D(output_channels=192,kernel_shape=(3,3,3),
                     name=end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point:return net,end_points
        end_point = 'MaxPool3d_3a_3x3'
        net = tf.nn.max_pool3d(net,ksize=(1,1,3,3,1),strides=(1,1,2,2,1),
                               padding=snt.SAME,name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point:return net,end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=64,kernel_shape=(1,1,1),
                                  name='Conv3d_0a_1x1')(net,is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=96,kernel_shape=(1,1,1),
                                  name='Conv3d_0a_1x1')(net,is_training=is_training)
                branch_1 = Unit3D(output_channels=128,kernel_shape=(3,3,3),
                                  name='Conv3d_0b_3x3')(branch_1,is_training=is_training)

            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=16,kernel_shape=(1,1,1),
                                  name='Conv3d_0a_1x1')(net,is_training=is_training)
                branch_2 = Unit3D(output_channels=32,kernel_shape=(3,3,3),
                                  name='Conv3d_0b_3x3')(branch_2,is_training=is_training)

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net,ksize=(1,1,3,3,1),
                                            strides=(1,1,1,1,1),padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=32,kernel_shape=(1,1,1),
                                  name='Conv3d_0b_1x1')(branch_3,is_training=is_training)

            net = tf.concat([branch_0,branch_1,branch_2,branch_3],4)
            end_points[end_point] = net
            if self._final_endpoint == end_point:return net,end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=96, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool3d_4a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                               padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=208, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=48, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=224, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=256, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=144, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=288, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool3d_5a_2x2'
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                               padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0a_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=384, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Logits'
        with tf.variable_scope(end_point):
            frame = net.shape.as_list()[1]
            height = net.shape.as_list()[2]
            width = net.shape.as_list()[3]
            net = tf.nn.avg_pool3d(net,ksize=(1,2,7,7,1),
                                   strides=(1,1,1,1,1),padding=snt.VALID)
            net = tf.contrib.layers.dropout(net,keep_prob=dropout_keep_prob,is_training=is_training)
            logits = Unit3D(output_channels=self._num_classes,
                            kernel_shape=[1,1,1],
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='Conv3d_0c_1x1')(net,is_training=is_training)
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
    rgb_input = tf.placeholder (tf.float32,
                                shape=(1, 16, 224, 224, 3))
    with tf.variable_scope ('RGB'):
        rgb_model = InceptionI3d (
            1001, spatia_squeeze=True, final_endpoint='Logits'
        )
        rgb_logits, _ = rgb_model (
            rgb_input, is_training=False, dropout_keep_prob=1.0
        )

    saver = tf.train.Saver(reshape=True)

    sess = tf.Session ()
    sess.run (tf.global_variables_initializer ())
    ckpt = tf.train.get_checkpoint_state(r'E:\action_recognition\data/checkpoints/rgb_imagenet/')
    # if ckpt is not None:
    saver.restore (sess, r'E:/action_recognition/data/checkpoints/rgb_imagenet/model.ckpt')


    # print(InceptionI3d.VALID_ENDPOINTS)
    # for i in tf.global_variables():
    #     print(i)