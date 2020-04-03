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
    def  __init__(self,
                 output_channels ,
                 kernel_shape = (1,1),
                 stride = (1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 name = 'unit_2d'):
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
                         padding=snt.SAME,initializers=initializer,
                         use_bias=self.use_bias,regularizers=regularizers)(inputs)
        if self.use_batch_norm:
            bn = snt.BatchNorm(scale=True)
            net = bn(net,is_training=is_training,test_local_stats=False)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        return net

class SE_bottleneck(snt.AbstractModule):

    def __init__(self,depth,
               depth_bottleneck,
               stride = (1,1),
               use_batch_norm = True,
               use_bias = False,
               name = 'bottleneck_v1'):
        super(SE_bottleneck, self).__init__(name=name)
        self._depth = depth
        self._depth_bottleneck = depth_bottleneck
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias

    def SE_block(self,inputs,ratio):
        heights = inputs.shape.as_list()[1]
        width = inputs.shape.as_list()[2]
        squeeze = tf.nn.avg_pool(inputs,ksize=(1,heights,width,1),strides=(1,1,1,1),padding='VALID')
        excitation = Unit2d(output_channels=self._depth // ratio,use_batch_norm=False,
                            use_bias=True,name='fc1')(squeeze,is_training=False)
        excitation = Unit2d(output_channels=self._depth ,use_batch_norm=False,activation_fn=None,
                            use_bias=True,name='fc2')(excitation,is_training=False)
        return tf.nn.sigmoid(excitation)

    def _build(self, inputs,is_training,ratio=16):
        depth_in = inputs.shape.as_list()[-1]
        if depth_in == self._depth:
            shortcut = inputs
        else:
            shortcut = Unit2d(output_channels=self._depth,kernel_shape=(1,1),
                              stride=self._stride,activation_fn=None,name='shortcut')(inputs,is_training=is_training)

        residual = Unit2d(output_channels=self._depth_bottleneck,kernel_shape=(1,1),stride=self._stride,
                          name='conv1')(inputs,is_training=is_training)
        residual = Unit2d (output_channels=self._depth_bottleneck, kernel_shape=(3, 3), stride=(1,1),
                           name='conv2') (residual, is_training=is_training)
        residual = Unit2d (output_channels=self._depth, kernel_shape=(1, 1), stride=(1, 1),
                           name='conv3',activation_fn=None) (residual, is_training=is_training)

        se_output = self.SE_block(residual,ratio=16)
        res_out = residual * se_output
        res_out = tf.nn.relu(res_out + shortcut)

        return res_out

class SE_Resnet(snt.AbstractModule):

    VALID_ENDPOINTS = (
        'conv1',
        'pool1',
        'block1',
        'block2',
        'block3',
        'block4',
        'logits',
        'Predictions'
    )

    def  __init__(self,num_classes = 1000,spatia_squeeze = True,unit_num = [3,4,6,3],
                 final_endpoint = 'logits',name = 'resnet_v1'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(SE_Resnet, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint
        self._unit_num = unit_num


    def _build(self, inputs ,is_training ,dropout_keep_prob = 0.5):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        net = inputs
        end_points = {}
        end_point = 'conv1'
        net = Unit2d(output_channels=64,kernel_shape=[7,7],
                     stride=[2,2],name = end_point)(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        end_point = 'pool1'
        net = tf.nn.max_pool(net,ksize=(1,3,3,1),strides=(1,2,2,1),
                               padding=snt.SAME,name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'block1'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[0]
            for i in range(num_units):
                with tf.variable_scope('unit_%d' % (i+1)):
                    if i != 0:
                        net = SE_bottleneck(depth=256,depth_bottleneck=64,stride=1)(net,is_training=is_training)
                    else:
                        net = SE_bottleneck(depth=256,depth_bottleneck=64,stride=1)(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points

        end_point = 'block2'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[1]
            for i in range (num_units):
                with tf.variable_scope ('unit_%d' % (i + 1)):
                    if i != 0:
                        net = SE_bottleneck (depth=512, depth_bottleneck=128, stride=1) (net, is_training=is_training)
                    else:
                        net = SE_bottleneck (depth=512, depth_bottleneck=128, stride=2) (net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'block3'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[2]
            for i in range (num_units):
                with tf.variable_scope ('unit_%d' % (i + 1)):
                    if i != 0:
                        net = SE_bottleneck (depth=1024, depth_bottleneck=256, stride=1) (net, is_training=is_training)
                    else:
                        net = SE_bottleneck (depth=1024, depth_bottleneck=256, stride=2) (net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'block4'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[3]
            for i in range (num_units):
                with tf.variable_scope ('unit_%d' % (i + 1)):
                    if i != 0:
                        net = SE_bottleneck (depth=2048, depth_bottleneck=512, stride=1) (net, is_training=is_training)
                    else:
                        net = SE_bottleneck (depth=2048, depth_bottleneck=512, stride=2) (net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'logits'
        height = net.shape.as_list()[1]
        width = net.shape.as_list()[2]
        net = tf.nn.avg_pool(net,ksize=(1,height,width,1),
                               strides=(1,1,1,1),padding=snt.VALID)
        net = tf.contrib.layers.dropout(net,keep_prob=dropout_keep_prob,is_training=is_training)
        logits = Unit2d(output_channels=self._num_classes,
                        kernel_shape=[1,1],
                        activation_fn=None,
                        use_batch_norm=False,
                        use_bias=True,
                        name=end_point)(net,is_training=is_training)
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
    # k = Unit2d(output_channels=3)
    inputs = tf.placeholder(tf.float32,[None,224,224,3])
    with tf.variable_scope('RGB'):
        Resnet_v1 = SE_Resnet(num_classes=101,final_endpoint='logits',name='resnet_v1_50')
        logits , e = Resnet_v1(inputs,is_training=True)
        rgb_var_map = {}
        for var in tf.global_variables ():
            if var.name.split ('/')[0] == 'RGB' :
                rgb_var_map[var.name.replace (':0', '')] = var
        rgb_saver = tf.train.Saver (var_list=rgb_var_map, reshape=True)

    for i in tf.global_variables():
        print(i)
