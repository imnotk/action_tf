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
                 name = 'unit_2d',
                 eval_type = 'rgb',
                 bn_trainable = True):
        super(Unit2d, self).__init__(name=name)
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        self.eval_type = eval_type
        self.use_pbn = bn_trainable

    def _build(self,inputs,is_training):
        net = snt.Conv2D(output_channels=self.output_channels,
                         kernel_shape=self.kernel_shape,
                         stride=self.stride,
                         padding=snt.SAME,initializers=initializer,
                         use_bias=self.use_bias,regularizers=regularizers)(inputs)
        if self.use_batch_norm:
                bn = snt.BatchNormV2(scale=True,update_ops_collection=tf.GraphKeys.UPDATE_OPS)
                net = bn(net,is_training=is_training,test_local_stats=False)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        return net

class bottleneck(snt.AbstractModule):

    def __init__(self,depth,
               depth_bottleneck,
               stride = (1,1),
               use_batch_norm = True,
               use_bias = False,
               name = 'bottleneck_v1',
               eval_type='rgb',
               bn_trainable=True):
        super(bottleneck, self).__init__(name=name)
        self._depth = depth
        self._depth_bottleneck = depth_bottleneck
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._eval_type = eval_type
        self.use_pbn = bn_trainable

    def _build(self, shortcut_inputs,residual_inputs,is_training):
        depth_in = shortcut_inputs.shape.as_list()[-1]
        if depth_in == self._depth:
            shortcut = shortcut_inputs
        else:
            shortcut = Unit2d(output_channels=self._depth,kernel_shape=(1,1),stride=self._stride,activation_fn=None,name='shortcut')(shortcut_inputs,is_training=is_training & self.use_pbn)

        residual = Unit2d(output_channels=self._depth_bottleneck,kernel_shape=(1,1),stride=self._stride,name='conv1')(residual_inputs,is_training=is_training & self.use_pbn)
        residual = Unit2d (output_channels=self._depth_bottleneck, kernel_shape=(3, 3), stride=(1,1),name='conv2') (residual,           is_training=is_training & self.use_pbn)
        residual = Unit2d (output_channels=self._depth, kernel_shape=(1, 1), stride=(1, 1),name='conv3',activation_fn=None) (residual, is_training=is_training & self.use_pbn)

        return tf.nn.relu(residual + shortcut)


def non_local(rgb_inputs,flow_inputs):
    if rgb_inputs.shape.as_list() != flow_inputs.shape.as_list():
        raise ValueError('not using the same feature map size')
    
    rgb_inputs_re = tf.reshape(rgb_inputs,[-1,rgb_inputs.shape.as_list()[1] * rgb_inputs.shape.as_list()[2],rgb_inputs.shape.as_list()[3]])
    flow_inputs_re = tf.reshape(flow_inputs,[-1,flow_inputs.shape.as_list()[1] * flow_inputs.shape.as_list()[2],flow_inputs.shape.as_list()[3]])

    rgb_flow = tf.matmul(rgb_inputs_re,flow_inputs_re,transpose_b = True)
    rgb_flow_act = tf.nn.softmax(rgb_flow)
    rgb_non_local = tf.matmul(rgb_flow_act,flow_inputs_re)
    rgb_non_local_re = tf.reshape(rgb_non_local,[-1,rgb_inputs.shape.as_list()[1] , rgb_inputs.shape.as_list()[2],rgb_inputs.shape.as_list()[3]])
    rgb_out = rgb_inputs + rgb_non_local_re
    return tf.nn.relu(rgb_out)

class Resnet(snt.AbstractModule):

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

    def  __init__(self,num_classes = 1000,spatia_squeeze = True,eval_type='rgb',unit_num = [3,4,6,3],use_pbn=False,
                 final_endpoint = 'logits',name = 'resnet_v1',inject_mode = 'multi'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(Resnet, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint
        self._eval_type = eval_type
        self._unit_num = unit_num
        self._inject_mode = inject_mode
        self.use_pbn = use_pbn

    def _build(self, inputs ,is_training ,dropout_keep_prob = 0.5,inject_block=None):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        if self.use_pbn:
            _use_bn = False
        else:
            _use_bn = True


        # we only need inject block feature size of 56,28,14,7
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
        _inject_block = []
        with tf.variable_scope(end_point):
            num_units = self._unit_num[0]
            for i in range(num_units):
                with tf.variable_scope('unit_%d' % (i+1)):
                    if i == 0:
                        net = bottleneck(depth=256,depth_bottleneck=64,stride=1)(net,net,is_training=is_training)
                        _inject_block.append(net)
                        if inject_block is not None:
                            if self._inject_mode == 'multi':
                                residual_net = net * inject_block[0]
                            elif self._inject_mode == 'add':
                                residual_net = net + inject_block[0]
                            elif self._inject_mode == 'non_local':
                                residual_net = non_local(net,inject_block[0])
                        else:
                            residual_net = net
                    else:
                        net = bottleneck(depth=256,depth_bottleneck=64,stride=1)(net,residual_net,is_training=is_training)
                        net = net
                        residual_net = net
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points

        end_point = 'block2'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[1]
            for i in range (num_units):
                with tf.variable_scope ('unit_%d' % (i + 1)):
                    if i != 0:
                        net = bottleneck (depth=512, depth_bottleneck=128, stride=1 ) (net,residual_net, is_training=is_training)
                        net = net
                        residual_net = net
                    else:
                        net = bottleneck (depth=512, depth_bottleneck=128, stride=2 ) (net,net,is_training=is_training)
                        _inject_block.append(net)
                        if inject_block is not None:
                            if self._inject_mode == 'multi':
                                residual_net = net * inject_block[1]
                            elif self._inject_mode == 'add':
                                residual_net = net + inject_block[1]
                            elif self._inject_mode == 'non_local':
                                residual_net = non_local(net,inject_block[1])
                        else:
                            residual_net = net
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'block3'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[2]
            for i in range (num_units):
                with tf.variable_scope ('unit_%d' % (i + 1)):
                    if i != 0:
                        net = bottleneck (depth=1024, depth_bottleneck=256, stride=1 ) (net,residual_net, is_training=is_training)
                        net = net
                        residual_net = net
                    else:
                        net = bottleneck (depth=1024, depth_bottleneck=256, stride=2 ) (net,net, is_training=is_training)
                        _inject_block.append(net)
                        if inject_block is not None:
                            if self._inject_mode == 'multi':
                                residual_net = net * inject_block[2]
                            elif self._inject_mode == 'add':
                                residual_net = net + inject_block[2]
                            elif self._inject_mode == 'non_local':
                                residual_net = non_local(net,inject_block[2])
                        else:
                            residual_net = net
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'block4'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[3]
            for i in range (num_units):
                with tf.variable_scope ('unit_%d' % (i + 1)):
                    if i != 0:
                        net = bottleneck (depth=2048, depth_bottleneck=512, stride=1 ) (net,residual_net, is_training=is_training)
                        net = net
                        residual_net = net
                    else:
                        net = bottleneck (depth=2048, depth_bottleneck=512, stride=2 ) (net,net,is_training=is_training)
                        _inject_block.append(net)
                        if inject_block is not None:
                            if self._inject_mode == 'multi':
                                residual_net = net * inject_block[3]
                            elif self._inject_mode == 'add':
                                residual_net = net + inject_block[3]
                            elif self._inject_mode == 'non_local':
                                residual_net = non_local(net,inject_block[3])
                        else:
                            residual_net = net
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
        
        end_points['inject_block'] = _inject_block
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


def inject_two_stream(rgb_inputs,flow_inputs,is_training,rgb_dr,flow_dr,num_classes):

    with tf.variable_scope('Flow'):
        flow_model = Resnet (num_classes=num_classes, name='resnet_v1_50')

    with tf.variable_scope('RGB'):
        rgb_model = Resnet (num_classes=num_classes, name='resnet_v1_50',inject_mode='non_local')
    
    flow_logits , flow_endpoints = flow_model(flow_inputs,is_training=is_training)

    inject_block = flow_endpoints['inject_block']

    rgb_logits , rgb_endpoints = rgb_model(rgb_inputs,is_training=is_training,inject_block=inject_block)

    for i in tf.global_variables():
        print(i)

if __name__ == "__main__":
    rgb_inputs = tf.placeholder(tf.float32,[None,224,224,3])
    flow_inputs = tf.placeholder(tf.float32,[None,224,224,3])
    inject_two_stream(rgb_inputs,flow_inputs,True,1,1,101)
    sess = tf.Session()
    train_writer = tf.summary.FileWriter ('./model/test', sess.graph)

    
