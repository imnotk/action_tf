from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt

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
                 use_bias = True,
                 name = 'conv_2d'):
        super(Unit2d, self).__init__(name=name)
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias

    def _build(self,inputs,is_training):
        net = snt.Conv2D(output_channels=self.output_channels,
                            kernel_shape=self.kernel_shape,
                            stride=self.stride,
                            padding=snt.SAME,
                            use_bias=self.use_bias,regularizers=regularizers)(inputs)
        if self.use_batch_norm:
            bn = snt.BatchNormV2(scale=True)
            net = bn(net,is_training=is_training,test_local_stats=False)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        return net


def pool(net, pool_type='avg', kernel=3, stride=1, padding=0):
  if pool_type == 'avg':
    fn = tf.nn.avg_pool
  elif pool_type == 'max':
    fn = tf.nn.max_pool
  else:
    raise ValueError('Unknown pool type')
  with tf.name_scope('%s_pool' % pool_type):
    net = fn(net, [1 , kernel, kernel, 1], strides=stride,
             padding='VALID' if padding==0 else 'SAME')
  return net


def inception_module(net, is_training,small_module=False,
                     num_outputs=[64,64,64,32,64,96,96],
                     force_max_pool=False):
    all_nets = []
    if not small_module:
        net_1 = Unit2d(num_outputs[0], [1, 1],name='1x1')(net,is_training=is_training)
        all_nets.append(net_1)

    net_2 = Unit2d(num_outputs[1], [1, 1],name='3x3_reduce')(net,is_training=is_training)
    net_2 = Unit2d(num_outputs[2], [3, 3],name='3x3',
                    stride=[2,2] if small_module else 1)(net_2,is_training=is_training)
    all_nets.append(net_2)

    net_3 = Unit2d(num_outputs[4], [1, 1],name='double_3x3_reduce')(net,is_training=is_training)
    net_3 = Unit2d(num_outputs[5], [3, 3],name='double_3x3_1')(net_3,is_training=is_training)
    net_3 = Unit2d(num_outputs[6], [3, 3],stride=[2,2] if small_module else 1,name='double_3x3_2')(net_3,is_training=is_training)
    all_nets.append(net_3)

    with tf.variable_scope('pool'):
        if small_module:
            net_4 = pool(net, 'max', 3, [1,2,2,1], 1)
        elif force_max_pool:
            net_4 = pool(net, 'max', 3, [1,1,1,1], 1)
        else:
            net_4 = pool(net, 'avg', 3, [1,1,1,1], 1)
    if not small_module:
        net_4 = Unit2d(num_outputs[3], [1, 1],name='pool_proj')(net_4,is_training=is_training)
    all_nets.append(net_4)

    net = tf.concat(all_nets,axis=-1)
    return net


class BNInception(snt.AbstractModule):

    VALID_ENDPOINTS = (
        'conv1/7x7_s2',
        'pool1/3x3_s2',
        'conv2/3x3_reduce',
        'conv2/3x3',
        'pool2/3x3_s2',
        'inception_3a',
        'inception_3b',
        'inception_3c',
        'inception_4a',
        'inception_4b',
        'inception_4c',
        'inception_4d',
        'inception_4e',
        'inception_5a',
        'inception_5b',
        'Logits',
        'Predictions',
    )

    def __init__(self,num_classes = 101,spatia_squeeze = True,frame_counts = 1,
                 final_endpoint = 'Logits',name = 'BNInception'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(BNInception, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint
        self._frame_counts = frame_counts


    def _build(self, inputs ,is_training ,dropout_keep_prob = 1.0):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        if len(inputs.shape.as_list()) == 5:
            H,W,C = inputs.shape.as_list()[2:]
            inputs = tf.reshape(inputs,[-1,H,W,C])

        final_endpoint = self._final_endpoint
        end_points = {}
        end_point = 'conv1/7x7_s2'
        net = Unit2d(64, [7, 7],stride=[2,2],name=end_point)(inputs,is_training=is_training)
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        # 112 x 112 x 64
        end_point = 'pool1/3x3_s2'
        net = tf.nn.max_pool(net, [1,3,3,1], name=end_point,
                                strides=[1,2,2,1], padding='SAME')
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        # 56 x 56 x 64
        end_point = 'conv2/3x3_reduce'
        net = Unit2d(64, [1, 1],name=end_point)(net, is_training )
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        end_point = 'conv2/3x3'
        net = Unit2d(192, [3, 3],name=end_point)(net, is_training )
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        end_point = 'pool2/3x3_s2'
        net = tf.nn.max_pool(net, [1,3,3,1], name=end_point, strides=[1,2,2,1],
                                padding='SAME')
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        # Inception module.
        end_point = 'inception_3a'
        with tf.variable_scope(end_point):
            net = inception_module(net, is_training )
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        end_point = 'inception_3b'
        with tf.variable_scope(end_point):
            net = inception_module(net, is_training , num_outputs=[64,64,96,64,64,96,96])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        end_point = 'inception_3c'
        with tf.variable_scope(end_point):
            net = inception_module(net, is_training , small_module=True,
                                num_outputs=[-1,128,160,-1,64,96,96])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        end_point = 'inception_4a'
        with tf.variable_scope(end_point):
            net = inception_module(net, is_training , num_outputs=[224,64,96,128,96,128,128])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        end_point = 'inception_4b'
        with tf.variable_scope(end_point):
            net = inception_module(net, is_training , num_outputs=[192,96,128,128,96,128,128])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        end_point = 'inception_4c'
        with tf.variable_scope(end_point):
            net = inception_module(net, is_training , num_outputs=[160,128,160,128,128,160,160])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        end_point = 'inception_4d'
        with tf.variable_scope(end_point):
            net = inception_module(net, is_training , num_outputs=[96,128,192,128,160,192,192])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        end_point = 'inception_4e'
        with tf.variable_scope(end_point):
            net = inception_module(net, is_training , small_module=True,
                                num_outputs=[-1,128,192,-1,192,256,256])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        end_point = 'inception_5a'
        with tf.variable_scope(end_point):
            net = inception_module(net, is_training , num_outputs=[352,192,320,128,160,224,224])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points

        end_point = 'inception_5b'
        with tf.variable_scope(end_point):
            net = inception_module(net, is_training , num_outputs=[352,192,320,128,192,224,224],
                                force_max_pool=True)
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
        
        end_point = 'Logits'
        net = tf.nn.avg_pool(net,ksize=(1,7,7,1),
                                strides=(1,1,1,1),padding=snt.VALID)
        net = tf.contrib.layers.dropout(net,keep_prob=dropout_keep_prob,is_training=is_training)
        logits = Unit2d(output_channels=self._num_classes,
                        kernel_shape=[1,1],
                        activation_fn=None,
                        use_batch_norm=False,name=end_point,
                        use_bias=True)(net,is_training=is_training)
        if self._spatia_squeeze:
            logits = tf.squeeze(logits,(1,2),name='SpatialSqueeze')
        logits = tf.reshape(logits,[-1, self._frame_counts, self._num_classes])
        averge_logits = tf.reduce_mean(logits,axis=1)

        end_points[end_point] = averge_logits
        if self._final_endpoint == end_point: return averge_logits, end_points
        

        end_point = 'Predictions'
        predictions = tf.nn.softmax(averge_logits)
        end_points[end_point] = predictions
        return predictions,end_points


if __name__ == "__main__":
    import numpy as np
    inputs = tf.placeholder(tf.float32,shape=(None,224,224,3))
    with tf.variable_scope('RGB'):
        logits ,e = BNInception(num_classes=1000)(inputs,is_training=True)
        saver = tf.train.Saver(reshape=True)
    sample = np.ones((1,224,224,3))
    for i in tf.trainable_variables():
        print(i)
    print('--------------------')
    print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))