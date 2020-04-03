from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import sonnet as snt

weight_initializer = {
    'w':tf.contrib.layers.xavier_initializer()
}

bias_initializer = {
    'b':tf.constant_initializer(value=0.2)
}

regularizers = {
    'w':tf.contrib.layers.l2_regularizer(scale=1.0)
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
        self.initializer = weight_initializer

    def _build(self,inputs,is_training):
        net = snt.Conv3D(output_channels=self.output_channels,
                         kernel_shape=self.kernel_shape,
                         stride=self.stride,
                         padding=snt.SAME,initializers=self.initializer,
                         use_bias=self.use_bias,regularizers=regularizers,name='conv_2d')(inputs)
        if self.use_batch_norm:
            bn = snt.BatchNormV2(scale=True)
            net = bn(net,is_training=is_training,test_local_stats=False)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        return net

class deep_bottleneck(snt.AbstractModule):

    def __init__(self,depth,
               depth_bottleneck,
               stride = (1,1,1),
               use_batch_norm = True,
               use_bias = False,
               name = 'bottleneck_v1'):
        super(deep_bottleneck, self).__init__(name=name)
        self._depth = depth
        self._depth_bottleneck = depth_bottleneck
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias

    def _build(self, inputs,is_training):
        depth_in = inputs.shape.as_list()[-1]
        if depth_in == self._depth and self._stride[1] == 1:
            shortcut = inputs
        else:
            shortcut = Unit3d(output_channels=self._depth,kernel_shape=(1,1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,
                              stride=self._stride,activation_fn=None,name='shortcut')(inputs,is_training=is_training)

        residual = Unit3d(output_channels=self._depth_bottleneck,kernel_shape=(1, 1, 1),stride=self._stride,use_batch_norm=self._use_batch_norm,
                          use_bias=self._use_bias,name='conv1')(inputs,is_training=is_training)
        residual = Unit3d (output_channels=self._depth_bottleneck, kernel_shape=(1, 3, 3), stride=(1, 1, 1),use_batch_norm=self._use_batch_norm,
                           use_bias=self._use_bias,name='conv2') (residual, is_training=is_training)
        residual = Unit3d (output_channels=self._depth, kernel_shape=(1, 1, 1), stride=(1, 1, 1),use_batch_norm=self._use_batch_norm,
                           use_bias=self._use_bias,name='conv3',activation_fn=None) (residual, is_training=is_training)

        return tf.nn.relu(residual + shortcut)

def _build_feature(total_endpoints):

        # we don't need the first two conv feature map due to low semantic information
        rgb_endpoints = total_endpoints[0]
        flow_endpoints = total_endpoints[1]
        feature_list = []

        rgb_56 = [];flow_56 = []
        rgb_28 = [];flow_28 = []
        rgb_14 = [];flow_14 = []
        rgb_7 = [];flow_7 = []

        rgb_endpoints_items = sorted(rgb_endpoints)
        flow_endpoints_items = sorted(flow_endpoints)
        for k  in rgb_endpoints_items:

            if 'pool' in k or 'logits' in k or 'Logits' in k or 'Pool' in k:
                continue

            v = rgb_endpoints[k]

            if v.shape.as_list()[2] == 56:
                rgb_56.append(v)
            if v.shape.as_list()[2] == 28:
                rgb_28.append(v)
            if v.shape.as_list()[2] == 14:
                rgb_14.append(v)
            if v.shape.as_list()[2] == 7:
                rgb_7.append(v)

        # rgb_56_feature = tf.concat(rgb_56,axis=-1)
        # rgb_28_feature = tf.concat(rgb_28,axis=-1)
        # rgb_14_feature = tf.concat(rgb_14,axis=-1)
        # rgb_7_feature = tf.concat(rgb_7,axis=-1)
        rgb_56_feature = rgb_56
        rgb_28_feature = rgb_28
        rgb_14_feature = rgb_14
        rgb_7_feature = rgb_7
        

        for k  in flow_endpoints_items:

            if 'pool' in k or 'logits' in k or 'Logits' in k or 'Pool' in k:
                continue
            
            v = flow_endpoints[k]

            if v.shape.as_list()[2] == 56:
                flow_56.append(v)
            if v.shape.as_list()[2] == 28:
                flow_28.append(v)
            if v.shape.as_list()[2] == 14:
                flow_14.append(v)
            if v.shape.as_list()[2] == 7:
                flow_7.append(v)
        
        # flow_56_feature = tf.concat(flow_56,axis=-1)
        # flow_28_feature = tf.concat(flow_28,axis=-1)
        # flow_14_feature = tf.concat(flow_14,axis=-1)
        # flow_7_feature = tf.concat(flow_7,axis=-1)
        flow_56_feature = flow_56
        flow_28_feature = flow_28
        flow_14_feature = flow_14
        flow_7_feature = flow_7


        feature_list.append([rgb_56_feature,flow_56_feature])
        feature_list.append([rgb_28_feature,flow_28_feature])
        feature_list.append([rgb_14_feature,flow_14_feature])
        feature_list.append([rgb_7_feature,flow_7_feature])
        return feature_list

class FeatureHierachyNetwork(snt.AbstractModule):

    def __init__(self,num_classes=101,middle_channels=256,spatial_squeeze=True,name='FHN',use_batch_norm=False,batch_size = None,use_bias=False,fusion_mode = 'add'):
        super (FeatureHierachyNetwork, self).__init__ (name=name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._middle_channels = middle_channels    
        self._batch_size = batch_size
        self._fusion_mode = fusion_mode


    def _build(self, feature_pairs,is_training=False,dropout_keep_prob=0.2):
        '''

        :param feature_pairs: dict (endpoint,(spatial_feature,motion_feature))
        :param is_training: batch norm parameter ,default False
        :param dropout_keep_prob: dropout parameter ,default 1.0
        :param simple_concat: simple concat for spatial and motion feature if True
        :return:
        '''
        # we only need feature size of 56,28,14,7

        if type(feature_pairs[0]) is dict:
            feature_list = _build_feature(feature_pairs)
        else:
            feature_list = feature_pairs
        
        feature_list = feature_list[1:]

        endpoints = {}
        feature_nums = len(feature_list)

        print(feature_list) 
        for i in range(feature_nums):
            if i == 0 :
                with tf.variable_scope('top_%d' % i):
                    feature_dimensions = self._middle_channels

                    rgb_feature = feature_list[i][0]
                    flow_feature = feature_list[i][1]

                    rgb_feature = tf.concat(rgb_feature,axis=-1)
                    flow_feature = tf.concat(flow_feature,axis=-1)

                    spatial_feature = Unit3d(output_channels=self._middle_channels,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,
                                                name='spatial')(rgb_feature,is_training=is_training)
                    motion_feature  = Unit3d(output_channels=self._middle_channels,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,
                                            name='motion')(flow_feature,is_training=is_training)
                    
                    if self._fusion_mode == 'add':
                        mixed_feature = tf.nn.relu(motion_feature + spatial_feature)
                        mixed_feature = tf.contrib.layers.dropout(mixed_feature,keep_prob=dropout_keep_prob,is_training=is_training)
                    elif self._fusion_mode == 'concat':
                        mixed_feature = tf.nn.relu(tf.concat([spatial_feature,motion_feature],axis=-1))
                        mixed_feature = tf.contrib.layers.dropout(mixed_feature,keep_prob=dropout_keep_prob,is_training=is_training)
                        
                    mixed_feature = Unit3d(output_channels=feature_dimensions // 4,kernel_shape=(1,7,7),stride=(1,2,2),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias)(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_1')(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_2')(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_3')(mixed_feature,is_training=is_training)

                    top_net = mixed_feature

                    endpoints['top_%d' % i] = top_net
            else:
                with tf.variable_scope('top_%d' % i):
                    feature_dimensions = self._middle_channels
                    
                    rgb_feature = feature_list[i][0]
                    flow_feature = feature_list[i][1]

                    rgb_feature = tf.concat(rgb_feature,axis=-1)
                    flow_feature = tf.concat(flow_feature,axis=-1)

                    spatial_feature = Unit3d(output_channels=self._middle_channels,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,
                                                name='spatial')(rgb_feature,is_training=is_training)
                    motion_feature  = Unit3d(output_channels=self._middle_channels,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,
                                            name='motion')(flow_feature,is_training=is_training)

                    if self._fusion_mode == 'add':
                        mixed_two_feature = tf.nn.relu(motion_feature + spatial_feature + top_net)
                        mixed_two_feature = tf.contrib.layers.dropout(mixed_two_feature,keep_prob=dropout_keep_prob,is_training=is_training)
                    elif self._fusion_mode == 'concat':
                        mixed_two_feature = tf.nn.relu(tf.concat([spatial_feature,motion_feature,top_net],axis=-1))
                        mixed_two_feature = tf.contrib.layers.dropout(mixed_two_feature,keep_prob=dropout_keep_prob,is_training=is_training)

                    if i != feature_nums - 1:
                        mixed_two_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,2,2),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_1')(mixed_two_feature,is_training=is_training)
                        top_net = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_2')(mixed_two_feature,is_training=is_training)
                    else:
                        # mixed_two_feature = Unit3d(output_channels=feature_dimensions * 4,kernel_shape=(1,1,1),stride=(1,1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias)(mixed_two_feature,is_training=is_training)
                        mixed_two_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_1')(mixed_two_feature,is_training=is_training)
                        top_net = mixed_two_feature
                    
                    endpoints['top_%d' % i] = top_net

        with tf.variable_scope('logits'):
            height = top_net.shape.as_list()[2]
            width = top_net.shape.as_list()[3]
            net = tf.nn.avg_pool3d(top_net,[1,1,height,width,1],[1,1,1,1,1],padding=snt.VALID)
            net = tf.contrib.layers.dropout(net,keep_prob=dropout_keep_prob,is_training=is_training)
            logits = Unit3d (output_channels=self._num_classes,
                            kernel_shape=[1, 1, 1],
                            activation_fn=None,
                            use_batch_norm= False,
                            use_bias= True
                            ) (net, is_training=is_training)
            if self._spatial_squeeze:
                logits = tf.squeeze (logits, (2, 3), name='SpatialSqueeze')
            logits = tf.reduce_mean(logits,axis=1)
        
        endpoints['logits'] = logits

        return logits , endpoints



if __name__ == "__main__":
    try:
        from model import  TS_resnet
    except:
        import TS_resnet
    rgb_input = tf.placeholder (tf.float32,
                                            [None,3, 224, 224, 3])
        
    flow_input = tf.placeholder (tf.float32,
                                        [None,3, 224, 224, 20])

    rgb_logits, rgb_endpoints = TS_resnet.Resnet() (rgb_input, is_training=False,
                                                    dropout_keep_prob=1.0)

    flow_logits, flow_endpoints = TS_resnet.Resnet() (flow_input, is_training=False,
                                                dropout_keep_prob=1.0)

    feature_list = [rgb_endpoints,flow_endpoints]
    with tf.variable_scope('fusion'):
        fusion_logits, endpoints = FeatureHierachyNetwork(fusion_mode='add') (feature_list, is_training=True, dropout_keep_prob=1.0)
    
    model_logits = fusion_logits
    print(model_logits)
    for i in tf.global_variables('fusion'):
        print(i)
