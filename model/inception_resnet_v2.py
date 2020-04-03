from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

initializer = {
    'w':tf.contrib.layers.xavier_initializer()
}
regularizers = {
    'w':tf.contrib.layers.l2_regularizer(scale=4e-5)
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
                 padding= snt.SAME,
                 name = 'unit_2d',
                 bn_trainable=True):
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
        self.padding = padding

    def _build(self,inputs,is_training,depthwise=False):
        if depthwise:
            net = snt.SeparableConv2D(output_channels=self.output_channels,
                                      channel_multiplier=8,
                                      kernel_shape=self.kernel_shape,
                                      stride=self.stride,padding=self.padding,
                                      use_bias=self.use_bias)(inputs)
        else:
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

def block35(net,is_training ,scale=1.0, activation_fn=tf.nn.relu, scope=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35'):
    with tf.variable_scope('Branch_0'):
      tower_conv = Unit2d(32, 1, name='Conv2d_1x1')(net,is_training=is_training)
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = Unit2d( 32, 1, name='Conv2d_0a_1x1')(net,is_training=is_training)
      tower_conv1_1 = Unit2d(32, 3, name='Conv2d_0b_3x3')(tower_conv1_0, is_training=is_training)
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = Unit2d(32, 1, name='Conv2d_0a_1x1')(net,is_training=is_training)
      tower_conv2_1 = Unit2d(48, 3, name='Conv2d_0b_3x3')(tower_conv2_0, is_training=is_training)
      tower_conv2_2 = Unit2d(64, 3, name='Conv2d_0c_3x3')(tower_conv2_1, is_training=is_training)
    mixed = tf.concat(axis=-1, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = Unit2d(net.get_shape()[-1], 1, use_batch_norm=False,use_bias=True,activation_fn=None, name='Conv2d_1x1')(mixed,is_training=is_training)
    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def block17(net, is_training,scale=1.0, activation_fn=tf.nn.relu, scope=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17'):
    with tf.variable_scope('Branch_0'):
      tower_conv = Unit2d(192, 1, name='Conv2d_1x1')(net,is_training=is_training)
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = Unit2d(128, 1, name='Conv2d_0a_1x1')(net,is_training=is_training)
      tower_conv1_1 = Unit2d(160, [1, 7],name='Conv2d_0b_1x7')(tower_conv1_0, is_training=is_training)
      tower_conv1_2 = Unit2d(192, [7, 1],name='Conv2d_0c_7x1')(tower_conv1_1, is_training=is_training)
    mixed = tf.concat(axis=-1, values=[tower_conv, tower_conv1_2])
    up = Unit2d(net.get_shape()[-1], 1, use_batch_norm=False,use_bias=True,activation_fn=None, name='Conv2d_1x1')(mixed, is_training=is_training)

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def block8(net, is_training,scale=1.0, activation_fn=tf.nn.relu, scope=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8'):
    with tf.variable_scope('Branch_0'):
      tower_conv = Unit2d(192, 1, name='Conv2d_1x1')(net,is_training=is_training)
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = Unit2d(192, 1, name='Conv2d_0a_1x1')(net,is_training=is_training)
      tower_conv1_1 = Unit2d(224, [1, 3], name='Conv2d_0b_1x3')(tower_conv1_0,is_training=is_training)
      tower_conv1_2 = Unit2d(256, [3, 1], name='Conv2d_0c_3x1')(tower_conv1_1,is_training=is_training)
    mixed = tf.concat(axis=-1, values=[tower_conv, tower_conv1_2])
    up = Unit2d(net.get_shape()[-1], 1, use_batch_norm=False,use_bias=True,activation_fn=None, name='Conv2d_1x1')(mixed,is_training=is_training)

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net

class InceptionResnetV2(snt.AbstractModule):

    VALID_ENDPOINTS = [
        'Conv2d_1a_3x3', 
        'Conv2d_2a_3x3', 
        'Conv2d_2b_3x3',
        'MaxPool_3a_3x3', 
        'Conv2d_3b_1x1', 
        'Conv2d_4a_3x3', 
        'MaxPool_5a_3x3',
        'Mixed_5b', 
        'Mixed_6a', 
        'PreAuxLogits', 
        'Mixed_7a', 
        'Conv2d_7b_1x1',
        'Logits',
        'Predictions',
    ]

    def __init__(self,num_classes = 101,spatia_squeeze = True,create_aux_logits = False,
                 final_endpoint = 'Logits',name = 'InceptionResnetV2'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionResnetV2, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint
        self._create_aux_logits = create_aux_logits

    def _build(self, inputs ,is_training ,dropout_keep_prob = 1.0):

        net = inputs
        end_points = {}
        padding = 'SAME'

        # 149 x 149 x 32
        end_point = 'Conv2d_1a_3x3'
        net = Unit2d(32, 3, stride=2, name='Conv2d_1a_3x3')(net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        # 147 x 147 x 32
        end_point = 'Conv2d_2a_3x3'
        net = Unit2d(32, 3, name='Conv2d_2a_3x3')(net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        # 147 x 147 x 64
        end_point = 'Conv2d_2b_3x3'
        net = Unit2d(64, 3, name='Conv2d_2b_3x3')(net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        # 73 x 73 x 64
        end_point = 'MaxPool_3a_3x3'
        net = tf.nn.max_pool(net, [1,3,3,1], strides=[1,2,2,1], padding=padding,
                                name='MaxPool_3a_3x3')
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        # 73 x 73 x 80
        end_point = 'Conv2d_3b_1x1'
        net = Unit2d(80, 1, name='Conv2d_3b_1x1')(net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        # 71 x 71 x 192
        end_point = 'Conv2d_4a_3x3'
        net = Unit2d(192, 3, name='Conv2d_4a_3x3')(net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        # 35 x 35 x 192
        end_point = 'MaxPool_5a_3x3'
        net = tf.nn.max_pool(net, [1,3,3,1], strides=[1,2,2,1], padding=padding,
                                name='MaxPool_5a_3x3')
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points

        # 35 x 35 x 320
        end_point = 'Mixed_5b'
        with tf.variable_scope('Mixed_5b'):
            with tf.variable_scope('Branch_0'):
                tower_conv = Unit2d(96, 1, name='Conv2d_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = Unit2d(48, 1, name='Conv2d_0a_1x1')(net, is_training=is_training)
                tower_conv1_1 = Unit2d(64, 5, name='Conv2d_0b_5x5')(tower_conv1_0, is_training=is_training)
            with tf.variable_scope('Branch_2'):
                tower_conv2_0 = Unit2d(64, 1, name='Conv2d_0a_1x1')(net, is_training=is_training)
                tower_conv2_1 = Unit2d(96, 3, name='Conv2d_0b_3x3')(tower_conv2_0, is_training=is_training)
                tower_conv2_2 = Unit2d(96, 3, name='Conv2d_0c_3x3')(tower_conv2_1, is_training=is_training)
            with tf.variable_scope('Branch_3'):
                tower_pool = tf.nn.avg_pool(net, [1,3,3,1], strides=[1,1,1,1], padding='SAME',
                                            name='AvgPool_0a_3x3')
                tower_pool_1 = Unit2d(64, 1,name='Conv2d_0b_1x1')(tower_pool, is_training=is_training)
            net = tf.concat(
                [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], -1)

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        # TODO(alemi): Register intermediate endpoints
        # net = slim.repeat(net, 10, block35, scale=0.17,
                            # activation_fn=activation_fn)
        with tf.variable_scope('Repeat'):
            for i in range(10):
                net = block35(net,is_training=is_training,scale=0.17,scope='block35_' + str(i+1))

        # 17 x 17 x 1088 if output_stride == 8,
        # 33 x 33 x 1088 if output_stride == 16
        end_point = 'Mixed_6a'
        with tf.variable_scope('Mixed_6a'):
            with tf.variable_scope('Branch_0'):
                tower_conv = Unit2d(384, 3, stride=2, name='Conv2d_1a_3x3')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = Unit2d(256, 1, name='Conv2d_0a_1x1')(net, is_training=is_training)
                tower_conv1_1 = Unit2d(256, 3, name='Conv2d_0b_3x3')(tower_conv1_0, is_training=is_training)
                tower_conv1_2 = Unit2d(384, 3,stride=2, name='Conv2d_1a_3x3')(tower_conv1_1,  is_training=is_training)
            with tf.variable_scope('Branch_2'):
                tower_pool = tf.nn.max_pool(net, [1,3,3,1], strides= [1,2,2,1],
                                            padding=padding,
                                            name='MaxPool_1a_3x3')
            net = tf.concat([tower_conv, tower_conv1_2, tower_pool], -1)

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        # TODO(alemi): register intermediate endpoints
        # with slim.arg_scope([Unit2d], rate=2 if use_atrous else 1):
        #     net = slim.repeat(net, 20, block17, scale=0.10,
        #                     activation_fn=activation_fn)
        with tf.variable_scope('Repeat_1'):
            for i in range(20):
                net = block17(net,is_training=is_training,scale=0.1 , scope='block17_' + str(i+1))
        end_points['PreAuxLogits'] = net

        # 8 x 8 x 2080
        end_point = 'Mixed_7a'
        with tf.variable_scope('Mixed_7a'):
            with tf.variable_scope('Branch_0'):
                tower_conv = Unit2d(256, 1, name='Conv2d_0a_1x1')(net,is_training=is_training)
                tower_conv_1 = Unit2d(384, 3, stride=2,
                                            name='Conv2d_1a_3x3')(tower_conv, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                tower_conv1 = Unit2d(256, 1,  name='Conv2d_0a_1x1')(net,is_training=is_training)
                tower_conv1_1 = Unit2d(288, 3, stride=2,
                                            name='Conv2d_1a_3x3')(tower_conv1, is_training=is_training)
            with tf.variable_scope('Branch_2'):
                tower_conv2 = Unit2d(256, 1, name='Conv2d_0a_1x1')(net,is_training=is_training)
                tower_conv2_1 = Unit2d(288, 3,name='Conv2d_0b_3x3')(tower_conv2, is_training=is_training)
                tower_conv2_2 = Unit2d(320, 3, stride=2,name='Conv2d_1a_3x3')(tower_conv2_1, is_training=is_training)
            with tf.variable_scope('Branch_3'):
                tower_pool = tf.nn.max_pool(net, [1,3,3,1], strides=[1,2,2,1],
                                            padding=padding,
                                            name='MaxPool_1a_3x3')
            net = tf.concat(
                [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], -1)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points

        # TODO(alemi): register intermediate endpoints
        # net = slim.repeat(net, 9, block8, scale=0.20, activation_fn=activation_fn)
        with tf.variable_scope('Repeat_2'):
            for i in range(9):
                net = block8(net,is_training=is_training,scale=0.2 ,scope = 'block8_' + str(i+1))
        net = block8(net, is_training=is_training,activation_fn=None)

        # 8 x 8 x 1536
        end_point = 'Conv2d_7b_1x1'
        net = Unit2d(1536, 1, name='Conv2d_7b_1x1')(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        
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
                            name='Logits')(net,is_training=is_training)
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
    import time 
    import numpy as np
    inputs = tf.placeholder(tf.float32,shape=(None,224,224,3))
    y_ = tf.placeholder(tf.float32,[None,101])
    with tf.variable_scope('RGB'):
        model_logits ,e = InceptionResnetV2(num_classes=101)(inputs,is_training=True)

    model_predictions = tf.nn.softmax (model_logits)
    cross_entropy = tf.reduce_mean (-tf.reduce_sum (y_ * tf.log (model_predictions + 1e-10),reduction_indices=[1]))

    update_ops = tf.get_collection (tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies (update_ops):
        train_op = tf.train.AdamOptimizer().minimize (cross_entropy)
    
    sample = np.ones((32,224,224,3))
    y = np.ones((32,101))
    feed_dict = {inputs:sample,y_:y}

    t = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_op,feed_dict=feed_dict)
    end_t = time.time()
    print('use time', end_t - t)