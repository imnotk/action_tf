from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt
slim = tf.contrib.slim


class vgg(snt.AbstractModule):

    VALID_ENDPOINTS = (
        'conv1',
        'conv2',
        'conv3',
        'conv4',
        'conv5',
        'fc6',
        'fc7',
        'fc8',
    )

    def  __init__(self,num_classes = 1000,spatia_squeeze = True,
                 final_endpoint = 'fc8',name = 'vgg_16'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(vgg, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint

    def _build(self,inputs,
               is_training=True,
               dropout_keep_prob=0.5):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ('endpoint not in VALID_ENDPOINTS')

        end_points = {}

        end_point = 'conv1'
        net = slim.repeat (
            inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d (net, [2, 2], scope='pool1')
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv2'
        net = slim.repeat (net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d (net, [2, 2], scope='pool2')
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv3'
        net = slim.repeat (net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d (net, [2, 2], scope='pool3')
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv4'
        net = slim.repeat (net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d (net, [2, 2], scope='pool4')
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv5'
        net = slim.repeat (net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d (net, [2, 2], scope='pool5')
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        # Use conv2d instead of fully_connected slim.
        end_point = 'fc6'
        net = slim.conv2d (net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout (
            net, dropout_keep_prob, is_training=is_training, scope='dropout6')
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'fc7'
        net = slim.conv2d (net, 4096, [1, 1], scope='fc7')
        net = slim.dropout (
            net, dropout_keep_prob, is_training=is_training, scope='dropout7')
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'fc8'
        net = slim.conv2d (
            net,
            self._num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='fc8')
        if self._spatial_squeeze:
            net = tf.squeeze (net, [1, 2])
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           endpoint = 'fc8'):

    VALID_ENDPOINTS = (
        'conv1',
        'conv2',
        'conv3',
        'conv4',
        'conv5',
        'fc6',
        'fc7',
        'fc8',
    )
    if endpoint not in VALID_ENDPOINTS:
        raise ('endpoint not in VALID_ENDPOINTS')

    with tf.variable_scope(scope):
        end_points = {}

        end_point = 'conv1'
        net = slim.repeat (
            inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d (net, [2, 2], scope='pool1')
        end_points[end_point] = net
        if endpoint == end_point: return net ,end_points

        end_point = 'conv2'
        net = slim.repeat (net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d (net, [2, 2], scope='pool2')
        end_points[end_point] = net
        if endpoint == end_point: return net, end_points

        end_point = 'conv3'
        net = slim.repeat (net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d (net, [2, 2], scope='pool3')
        end_points[end_point] = net
        if endpoint == end_point: return net, end_points

        end_point = 'conv4'
        net = slim.repeat (net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d (net, [2, 2], scope='pool4')
        end_points[end_point] = net
        if endpoint == end_point: return net, end_points

        end_point = 'conv5'
        net = slim.repeat (net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d (net, [2, 2], scope='pool5')
        end_points[end_point] = net
        if endpoint == end_point: return net, end_points

        # Use conv2d instead of fully_connected slim.
        end_point = 'fc6'
        net = slim.conv2d (net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout (
            net, dropout_keep_prob, is_training=is_training, scope='dropout6')
        end_points[end_point] = net
        if endpoint == end_point: return net, end_points

        end_point = 'fc7'
        net = slim.conv2d (net, 4096, [1, 1], scope='fc7')
        net = slim.dropout (
            net, dropout_keep_prob, is_training=is_training, scope='dropout7')
        end_points[end_point] = net
        if endpoint == end_point: return net, end_points

        end_point = 'fc8'
        net = slim.conv2d (
            net,
            num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='fc8')
        if spatial_squeeze:
            net = tf.squeeze(net,[1,2])
        end_points[end_point] = net
        if endpoint == end_point: return net, end_points








if __name__ == '__main__':
    # k = Unit2d(output_channels=3)
    inputs = tf.placeholder(tf.float32,[None,224,224,20])
    with tf.variable_scope('Flow'):
        vgg_16 = vgg(num_classes=1000)
        logits , e = vgg_16(inputs,is_training=True)
        rgb_var_map = {}
        for var in tf.global_variables ():
            if var.name.split('/')[0] == 'Flow':
                rgb_var_map[var.name.replace(':0','')] = var
        rgb_saver = tf.train.Saver(var_list=rgb_var_map)
    # for i in e:
    #     print(i,e[i])
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # ckpt = tf.train.get_checkpoint_state(r'E:\action_recognition\data\rgb_vgg_16_2016_08_28 ')
        # print(ckpt)
        # if ckpt is not None:
        rgb_saver.restore(sess,r'E:\action_recognition\data\flow_vgg_16_2016_08_28\flow_vgg_16.ckpt')
    # print(m(inputs,True))

    # print(InceptionV1.VALID_ENDPOINTS)
    #
    for var in tf.global_variables():
        print(var)