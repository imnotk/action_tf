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
                 bn_trainable = True):
        super(Unit2d, self).__init__(name=name)
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
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
               bn_trainable=True):
        super(bottleneck, self).__init__(name=name)
        self._depth = depth
        self._depth_bottleneck = depth_bottleneck
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self.use_pbn = bn_trainable

    def _build(self, inputs,is_training):
        depth_in = inputs.shape.as_list()[-1]
        if depth_in == self._depth:
            shortcut = inputs
        else:
            shortcut = Unit2d(output_channels=self._depth,kernel_shape=(1,1),stride=self._stride,activation_fn=None,name='shortcut')(inputs,is_training=is_training)

        residual = Unit2d(output_channels=self._depth_bottleneck,kernel_shape=(1,1),stride=self._stride,name='conv1')(inputs,is_training=is_training)
        residual = Unit2d (output_channels=self._depth_bottleneck, kernel_shape=(3,3), stride=(1,1),name='conv2') (residual,is_training=is_training)
        residual = Unit2d (output_channels=self._depth, kernel_shape=(1,1), stride=(1,1),name='conv3',activation_fn=None) (residual,is_training=is_training)

        return residual + shortcut

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

    def  __init__(self,num_classes = 1000,spatia_squeeze = True,unit_num = [3,4,6,3],
                 final_endpoint = 'logits',name = 'resnet_v1'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(Resnet, self).__init__(name = name)
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
                        net_no_relu = bottleneck(depth=256,depth_bottleneck=64,stride=1 )(net,is_training=is_training)
                        net = tf.nn.relu(net_no_relu)
                    else:
                        net_no_relu = bottleneck(depth=256,depth_bottleneck=64,stride=1 )(net,is_training=is_training)
                        net = tf.nn.relu(net_no_relu)
        end_points[end_point+'_no_relu'] = net_no_relu
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points

        end_point = 'block2'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[1]
            for i in range (num_units):
                with tf.variable_scope ('unit_%d' % (i + 1)):
                    if i != 0:
                        net_no_relu = bottleneck (depth=512, depth_bottleneck=128, stride=1 ) (net, is_training=is_training)
                        net = tf.nn.relu(net_no_relu)
                    else:
                        net_no_relu = bottleneck (depth=512, depth_bottleneck=128, stride=2 ) (net, is_training=is_training)
                        net = tf.nn.relu(net_no_relu)
        end_points[end_point+'_no_relu'] = net_no_relu
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'block3'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[2]
            for i in range (num_units):
                with tf.variable_scope ('unit_%d' % (i + 1)):
                    if i != 0:
                        net_no_relu = bottleneck (depth=1024, depth_bottleneck=256, stride=1 ) (net, is_training=is_training)
                        net = tf.nn.relu(net_no_relu)
                    else:
                        net_no_relu = bottleneck (depth=1024, depth_bottleneck=256, stride=2 ) (net, is_training=is_training)
                        net = tf.nn.relu(net_no_relu)
        end_points[end_point+'_no_relu'] = net_no_relu
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'block4'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[3]
            for i in range (num_units):
                with tf.variable_scope ('unit_%d' % (i + 1)):
                    if i != 0:
                        net_no_relu = bottleneck (depth=2048, depth_bottleneck=512, stride=1 ) (net, is_training=is_training)
                        net = tf.nn.relu(net_no_relu)
                    else:
                        net_no_relu = bottleneck (depth=2048, depth_bottleneck=512, stride=2 ) (net, is_training=is_training)
                        net = tf.nn.relu(net_no_relu)
        end_points[end_point+'_no_relu'] = net_no_relu
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
        
        with tf.variable_scope('Attention'):
            fm_1 , fm_2 , fm_3 , fm_4 = end_points['block1_no_relu'] , end_points['block2_no_relu'] , end_points['block3_no_relu'] , end_points['block4_no_relu']

            feature_map_list = [fm_1, fm_2, fm_3, fm_4]

            with tf.variable_scope('spatial'):
                # fm_4 = Unit2d(output_channels=256,kernel_shape=[1,1],activation_fn=None,use_batch_norm=False,use_bias=True)(fm_4,is_training=is_training)
                # fm_1 = tf.nn.max_pool(fm_1,[1,3,3,1],[1,2,2,1],padding=snt.SAME)
                # fm_1 = tf.nn.max_pool(fm_1,[1,3,3,1],[1,2,2,1],padding=snt.SAME)
                # fm_1 = tf.nn.max_pool(fm_1,[1,3,3,1],[1,2,2,1],padding=snt.SAME)
            
                # fm_2 = tf.nn.max_pool(fm_2,[1,3,3,1],[1,2,2,1],padding=snt.SAME)
                # fm_2 = tf.nn.max_pool(fm_2,[1,3,3,1],[1,2,2,1],padding=snt.SAME)

                # fm_3 = tf.nn.max_pool(fm_3,[1,3,3,1],[1,2,2,1],padding=snt.SAME)

                f_re = tf.reshape(fm_4,[-1,fm_4.shape.as_list()[3],fm_4.shape.as_list()[1] * fm_4.shape.as_list()[2]])

                w = tf.get_variable(shape=[fm_4.shape.as_list()[1] * fm_4.shape.as_list()[2],fm_4.shape.as_list()[3]],name='non_local_mat_w',initializer=tf.initializers.truncated_normal)
                b = tf.get_variable(shape=[fm_4.shape.as_list()[1] * fm_4.shape.as_list()[2]],name='non_local_mat_b')
                f_matmul = tf.einsum('jk,ikl->ijl',w,f_re)
                # w_exp = tf.tile(w,[fm_4.shape.as_list()[0],1,1])
                # f_matmul = tf.matmul(f_re,w_exp,transpose_b=True)
                f_matmul = tf.nn.bias_add(f_matmul , b)
                f_matmul = tf.nn.softmax(f_matmul)
                f_matmul = tf.nn.l2_normalize(f_matmul,axis=-1)
                f_non_local = tf.matmul(f_matmul,f_re,transpose_b=True)
                f_non_local_re = tf.reshape(f_non_local,[-1,fm_4.shape.as_list()[1], fm_4.shape.as_list()[2],fm_4.shape.as_list()[3]])
                f_final_spatial = f_non_local_re
                
                # f_final_spatial = tf.concat([fm_1,fm_2,fm_3,fm_4],axis=-1)

            # with tf.variable_scope('channel'):
            #     f_re = tf.reshape(fm_4,[-1,fm_4.shape.as_list()[1] * fm_4.shape.as_list()[2],fm_4.shape.as_list()[3]])
            #     f_matmul = tf.matmul(f_re,f_re,transpose_a=True)
            #     f_matmul = tf.nn.softmax(f_matmul)
            #     f_non_local = tf.matmul(f_matmul,f_re,transpose_b=True)
            #     f_non_local_re = tf.reshape(f_non_local,[-1,fm_4.shape.as_list()[1], fm_4.shape.as_list()[2],fm_4.shape.as_list()[3]])
            #     f_final_channel = fm_4 + f_non_local_re
            
            net = f_final_spatial
            # net = (f_final_channel + f_final_spatial) / 2

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

if __name__ == "__main__":
    inputs = tf.placeholder(tf.float32,[None,224,224,3])
    a = Resnet()(inputs,True)

    for i in tf.global_variables():
        print(i)


    
