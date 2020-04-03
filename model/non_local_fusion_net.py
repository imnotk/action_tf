from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import sonnet as snt

try:
    from fusion_net import Unit2D,_build_feature , deep_bottleneck
    from non_local_fusion_helper import space_nonlocal_fusion,channel_nonlocal_fusion
except:
    from model.fusion_net import Unit2D,_build_feature,deep_bottleneck
    from model.non_local_fusion_helper import space_nonlocal_fusion,channel_nonlocal_fusion

class space_cross_correlation_Network(snt.AbstractModule):

    def __init__(self,num_classes=101,middle_channels=256,spatial_squeeze=True,name='FHN',use_batch_norm=False,use_bias= False,batch_size = None,fusion_mode='add'):
        super (space_cross_correlation_Network, self).__init__ (name=name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._use_batch_norm = use_batch_norm
        self._middle_channels = middle_channels    
        self._batch_size = batch_size
        self._fusion_mode = fusion_mode
        self._use_bias = use_bias
        self.middle_channels = [128,256,256,256]
        self.reduce_dimensions = [256,256,256,256]

    def _build(self, feature_pairs,is_training=False,dropout_keep_prob=1.0):
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
        middle_channels = self.middle_channels[1:]
        reduce_dim = self.reduce_dimensions[1:]

        print(feature_list)

        endpoints = {}
        feature_nums = len(feature_list)
        for i in range(feature_nums):
            if i == 0 :
                with tf.variable_scope('top_%d' % i):
                    feature_dimensions = middle_channels[i]
                    reduce_dimensions = reduce_dim[i]

                    rgb_feature = feature_list[i][0]
                    flow_feature = feature_list[i][1]

                    rgb_feature = tf.concat(rgb_feature,axis=-1)
                    flow_feature = tf.concat(flow_feature,axis=-1)

                    rgb_feature = Unit2D(output_channels=reduce_dimensions,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,name='spatial')(rgb_feature,is_training=is_training)
                    flow_feature = Unit2D(output_channels=reduce_dimensions,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,name='motion')(flow_feature,is_training=is_training)

                    spatial_attention_feature = space_nonlocal_fusion(name='rgb_flow_fusion')(rgb_feature,flow_feature,is_training)
                    motion_attention_feature = space_nonlocal_fusion(name='flow_rgb_fusion')(flow_feature,rgb_feature,is_training)

                    spatial_attention_feature = rgb_feature + spatial_attention_feature
                    motion_attention_feature = flow_feature + motion_attention_feature

                    if self._fusion_mode == 'add':
                        mixed_feature = tf.nn.relu(motion_attention_feature + spatial_attention_feature)
                        mixed_feature = tf.contrib.layers.dropout(mixed_feature,keep_prob=dropout_keep_prob,is_training=is_training)
                    elif self._fusion_mode == 'concat':
                        mixed_feature  = tf.nn.relu(tf.concat([spatial_attention_feature,motion_attention_feature],axis=-1))
                        mixed_feature = tf.contrib.layers.dropout(mixed_feature,keep_prob=dropout_keep_prob,is_training=is_training)

                    mixed_feature = Unit2D(output_channels=feature_dimensions // 4,kernel_shape=(7,7),stride=(2,2),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias)(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_1')(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_2')(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_3')(mixed_feature,is_training=is_training)
                    top_net = mixed_feature
                    endpoints['top_%d' % i] = top_net
            else:
                with tf.variable_scope('top_%d' % i):
                    feature_dimensions = middle_channels[i]
                    reduce_dimensions = reduce_dim[i]
                    
                    rgb_feature = feature_list[i][0]
                    flow_feature = feature_list[i][1]

                    rgb_feature = tf.concat(rgb_feature,axis=-1)
                    flow_feature = tf.concat(flow_feature,axis=-1)

                    rgb_feature = Unit2D(output_channels=reduce_dimensions,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,name='spatial')(rgb_feature,is_training=is_training)
                    flow_feature = Unit2D(output_channels=reduce_dimensions,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,name='motion')(flow_feature,is_training=is_training)

                    spatial_attention_feature = space_nonlocal_fusion(name='rgb_flow_fusion')(rgb_feature,flow_feature,is_training)
                    motion_attention_feature = space_nonlocal_fusion(name='flow_rgb_fusion')(flow_feature,rgb_feature,is_training)
                    
                    spatial_attention_feature = rgb_feature + spatial_attention_feature
                    motion_attention_feature = flow_feature + motion_attention_feature

                    if self._fusion_mode == 'add':
                        mixed_two_feature = tf.nn.relu(motion_attention_feature + spatial_attention_feature + top_net)
                        mixed_two_feature = tf.contrib.layers.dropout(mixed_two_feature,keep_prob=dropout_keep_prob,is_training=is_training)
                    elif self._fusion_mode == 'concat':
                        mixed_two_feature = tf.nn.relu(tf.concat([spatial_attention_feature,motion_attention_feature,top_net],axis=-1))
                        mixed_two_feature = tf.contrib.layers.dropout(mixed_two_feature,keep_prob=dropout_keep_prob,is_training=is_training)

                    if i != feature_nums - 1:
                        mixed_two_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(2,2),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_1')(mixed_two_feature,is_training=is_training)
                        top_net = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_2')(mixed_two_feature,is_training=is_training)
                    else:
                        mixed_two_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_1')(mixed_two_feature,is_training=is_training)
                        top_net = mixed_two_feature

                    endpoints['top_%d' % i] = top_net



        with tf.variable_scope('logits'):
            height = top_net.shape.as_list()[1]
            width = top_net.shape.as_list()[2]
            net = tf.nn.avg_pool(top_net,[1,height,width,1],[1,1,1,1],padding=snt.VALID)
            net = tf.contrib.layers.dropout(net,keep_prob=dropout_keep_prob,is_training=is_training)
            logits = Unit2D (output_channels=self._num_classes,
                             kernel_shape=[1, 1],
                             activation_fn=None,
                             use_batch_norm= False,
                             use_bias= True
                             ) (net, is_training=is_training)
            if self._spatial_squeeze:
                logits = tf.squeeze (logits, (1, 2), name='SpatialSqueeze')
            endpoints['logits'] = logits

        return logits , endpoints

class channel_cross_correlation_Network(snt.AbstractModule):

    def __init__(self,num_classes=101,middle_channels=256,spatial_squeeze=True,name='FHN',use_batch_norm=False,use_bias= False,batch_size = None,fusion_mode='add'):
        super (channel_cross_correlation_Network, self).__init__ (name=name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._use_batch_norm = use_batch_norm
        self._middle_channels = middle_channels    
        self._batch_size = batch_size
        self._fusion_mode = fusion_mode
        self._use_bias = use_bias
        self.middle_channels = [128,256,256,256]


    def _build(self, feature_pairs,is_training=False,dropout_keep_prob=1.0):
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
        middle_channels = self.middle_channels[1:]

        print(feature_list)
        reduce_dimensions = self._middle_channels

        endpoints = {}
        feature_nums = len(feature_list)
        for i in range(feature_nums):
            if i == 0 :
                with tf.variable_scope('top_%d' % i):
                    feature_dimensions = middle_channels[i]

                    rgb_feature = feature_list[i][0]
                    flow_feature = feature_list[i][1]

                    rgb_feature = tf.concat(rgb_feature,axis=-1)
                    flow_feature = tf.concat(flow_feature,axis=-1)

                    rgb_feature = Unit2D(output_channels=reduce_dimensions,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,name='spatial')(rgb_feature,is_training=is_training)
                    flow_feature = Unit2D(output_channels=reduce_dimensions,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,name='motion')(flow_feature,is_training=is_training)

                    spatial_attention_feature = channel_nonlocal_fusion(name='rgb_flow_fusion')(rgb_feature,flow_feature,is_training)
                    motion_attention_feature = channel_nonlocal_fusion(name='flow_rgb_fusion')(flow_feature,rgb_feature,is_training)

                    spatial_attention_feature = rgb_feature + spatial_attention_feature
                    motion_attention_feature = flow_feature + motion_attention_feature

                    if self._fusion_mode == 'add':
                        mixed_feature = tf.nn.relu(motion_attention_feature + spatial_attention_feature)
                        mixed_feature = tf.contrib.layers.dropout(mixed_feature,keep_prob=dropout_keep_prob,is_training=is_training)
                    elif self._fusion_mode == 'concat':
                        mixed_feature  = tf.nn.relu(tf.concat([spatial_attention_feature,motion_attention_feature],axis=-1))
                        mixed_feature = tf.contrib.layers.dropout(mixed_feature,keep_prob=dropout_keep_prob,is_training=is_training)

                    mixed_feature = Unit2D(output_channels=feature_dimensions // 4,kernel_shape=(7,7),stride=(2,2),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias)(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_1')(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_2')(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_3')(mixed_feature,is_training=is_training)
                    top_net = mixed_feature
                    endpoints['top_%d' % i] = top_net
            else:
                with tf.variable_scope('top_%d' % i):
                    feature_dimensions = middle_channels[i]
                    
                    rgb_feature = feature_list[i][0]
                    flow_feature = feature_list[i][1]

                    rgb_feature = tf.concat(rgb_feature,axis=-1)
                    flow_feature = tf.concat(flow_feature,axis=-1)

                    rgb_feature = Unit2D(output_channels=reduce_dimensions,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,name='spatial')(rgb_feature,is_training=is_training)
                    flow_feature = Unit2D(output_channels=reduce_dimensions,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,name='motion')(flow_feature,is_training=is_training)

                    spatial_attention_feature = channel_nonlocal_fusion(name='rgb_flow_fusion')(rgb_feature,flow_feature,is_training)
                    motion_attention_feature = channel_nonlocal_fusion(name='flow_rgb_fusion')(flow_feature,rgb_feature,is_training)

                    spatial_attention_feature = rgb_feature + spatial_attention_feature
                    motion_attention_feature = flow_feature + motion_attention_feature

                    if self._fusion_mode == 'add':
                        mixed_two_feature = tf.nn.relu(motion_attention_feature + spatial_attention_feature + top_net)
                        mixed_two_feature = tf.contrib.layers.dropout(mixed_two_feature,keep_prob=dropout_keep_prob,is_training=is_training)
                    elif self._fusion_mode == 'concat':
                        mixed_two_feature = tf.nn.relu(tf.concat([spatial_attention_feature,motion_attention_feature,top_net],axis=-1))
                        mixed_two_feature = tf.contrib.layers.dropout(mixed_two_feature,keep_prob=dropout_keep_prob,is_training=is_training)

                    if i != feature_nums - 1:
                        mixed_two_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(2,2),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_1')(mixed_two_feature,is_training=is_training)
                        top_net = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_2')(mixed_two_feature,is_training=is_training)
                    else:
                        mixed_two_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_1')(mixed_two_feature,is_training=is_training)
                        top_net = mixed_two_feature

                    endpoints['top_%d' % i] = top_net



        with tf.variable_scope('logits'):
            height = top_net.shape.as_list()[1]
            width = top_net.shape.as_list()[2]
            net = tf.nn.avg_pool(top_net,[1,height,width,1],[1,1,1,1],padding=snt.VALID)
            net = tf.contrib.layers.dropout(net,keep_prob=dropout_keep_prob,is_training=is_training)
            logits = Unit2D (output_channels=self._num_classes,
                             kernel_shape=[1, 1],
                             activation_fn=None,
                             use_batch_norm= False,
                             use_bias= True
                             ) (net, is_training=is_training)
            if self._spatial_squeeze:
                logits = tf.squeeze (logits, (1, 2), name='SpatialSqueeze')
            endpoints['logits'] = logits

        return logits , endpoints

class cross_correlation_Network(snt.AbstractModule):

    def __init__(self,num_classes=101,middle_channels=256,spatial_squeeze=True,name='FHN',use_batch_norm=False,use_bias= False,batch_size = None,fusion_mode='add'):
        super (cross_correlation_Network, self).__init__ (name=name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._use_batch_norm = use_batch_norm
        self._middle_channels = middle_channels    
        self._batch_size = batch_size
        self._fusion_mode = fusion_mode
        self._use_bias = use_bias
        self.middle_channels = [128,256,256,256]
        self.reduce_dimensions = [128,256,256,256]


    def _build(self, feature_pairs,is_training=False,dropout_keep_prob=1.0):
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
        middle_channels = self.middle_channels[1:]
        reduce_dim = self.reduce_dimensions[1:]

        print(feature_list)

        endpoints = {}
        feature_nums = len(feature_list)
        for i in range(feature_nums):
            if i == 0 :
                with tf.variable_scope('top_%d' % i):
                    feature_dimensions = middle_channels[i]
                    reduce_dimensions = reduce_dim[i]

                    rgb_feature = feature_list[i][0]
                    flow_feature = feature_list[i][1]

                    rgb_feature = tf.concat(rgb_feature,axis=-1)
                    flow_feature = tf.concat(flow_feature,axis=-1)

                    rgb_feature = Unit2D(output_channels=reduce_dimensions,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,name='spatial')(rgb_feature,is_training=is_training)
                    flow_feature = Unit2D(output_channels=reduce_dimensions,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,name='motion')(flow_feature,is_training=is_training)

                    channel_spatial_attention_feature = channel_nonlocal_fusion(name='channel_rgb_flow_fusion')(rgb_feature,flow_feature,is_training)
                    channel_motion_attention_feature = channel_nonlocal_fusion(name='channel_flow_rgb_fusion')(flow_feature,rgb_feature,is_training)

                    space_spatial_attention_feature = space_nonlocal_fusion(name='space_rgb_flow_fusion')(rgb_feature,flow_feature,is_training)
                    space_motion_attention_feature = space_nonlocal_fusion(name='space_flow_rgb_fusion')(flow_feature,rgb_feature,is_training)

                    motion_attention_feature = rgb_feature + space_motion_attention_feature + channel_motion_attention_feature
                    spatial_attention_feature = flow_feature + space_spatial_attention_feature + channel_spatial_attention_feature

                    if self._fusion_mode == 'add':
                        mixed_feature = tf.nn.relu(motion_attention_feature + spatial_attention_feature)
                        mixed_feature = tf.contrib.layers.dropout(mixed_feature,keep_prob=dropout_keep_prob,is_training=is_training)
                    elif self._fusion_mode == 'concat':
                        mixed_feature  = tf.nn.relu(tf.concat([spatial_attention_feature,motion_attention_feature],axis=-1))
                        mixed_feature = tf.contrib.layers.dropout(mixed_feature,keep_prob=dropout_keep_prob,is_training=is_training)

                    mixed_feature = Unit2D(output_channels=feature_dimensions // 4,kernel_shape=(7,7),stride=(2,2),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias)(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_1')(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_2')(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_3')(mixed_feature,is_training=is_training)
                    top_net = mixed_feature
                    endpoints['top_%d' % i] = top_net
            else:
                with tf.variable_scope('top_%d' % i):
                    feature_dimensions = middle_channels[i]
                    reduce_dimensions = reduce_dim[i]
                    
                    rgb_feature = feature_list[i][0]
                    flow_feature = feature_list[i][1]

                    rgb_feature = tf.concat(rgb_feature,axis=-1)
                    flow_feature = tf.concat(flow_feature,axis=-1)

                    rgb_feature = Unit2D(output_channels=reduce_dimensions,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,name='spatial')(rgb_feature,is_training=is_training)
                    flow_feature = Unit2D(output_channels=reduce_dimensions,use_batch_norm=self._use_batch_norm,activation_fn=None,use_bias=self._use_bias,name='motion')(flow_feature,is_training=is_training)

                    channel_spatial_attention_feature = channel_nonlocal_fusion(name='channel_rgb_flow_fusion')(rgb_feature,flow_feature,is_training)
                    channel_motion_attention_feature = channel_nonlocal_fusion(name='channel_flow_rgb_fusion')(flow_feature,rgb_feature,is_training)

                    space_spatial_attention_feature = space_nonlocal_fusion(name='space_rgb_flow_fusion')(rgb_feature,flow_feature,is_training)
                    space_motion_attention_feature = space_nonlocal_fusion(name='space_flow_rgb_fusion')(flow_feature,rgb_feature,is_training)

                    motion_attention_feature = rgb_feature + space_motion_attention_feature + channel_motion_attention_feature
                    spatial_attention_feature = flow_feature + space_spatial_attention_feature + channel_spatial_attention_feature

                    if self._fusion_mode == 'add':
                        mixed_two_feature = tf.nn.relu(motion_attention_feature + spatial_attention_feature + top_net)
                        mixed_two_feature = tf.contrib.layers.dropout(mixed_two_feature,keep_prob=dropout_keep_prob,is_training=is_training)
                    elif self._fusion_mode == 'concat':
                        mixed_two_feature = tf.nn.relu(tf.concat([spatial_attention_feature,motion_attention_feature,top_net],axis=-1))
                        mixed_two_feature = tf.contrib.layers.dropout(mixed_two_feature,keep_prob=dropout_keep_prob,is_training=is_training)

                    if i != feature_nums - 1:
                        mixed_two_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(2,2),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_1')(mixed_two_feature,is_training=is_training)
                        top_net = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_2')(mixed_two_feature,is_training=is_training)
                    else:
                        mixed_two_feature = deep_bottleneck(depth=feature_dimensions,depth_bottleneck=feature_dimensions // 4,stride=(1,1),use_batch_norm=self._use_batch_norm,use_bias=self._use_bias,name='adaption_layer_1')(mixed_two_feature,is_training=is_training)
                        top_net = mixed_two_feature

                    endpoints['top_%d' % i] = top_net

        with tf.variable_scope('logits'):
            height = top_net.shape.as_list()[1]
            width = top_net.shape.as_list()[2]
            net = tf.nn.avg_pool(top_net,[1,height,width,1],[1,1,1,1],padding=snt.VALID)
            net = tf.contrib.layers.dropout(net,keep_prob=dropout_keep_prob,is_training=is_training)
            logits = Unit2D (output_channels=self._num_classes,
                             kernel_shape=[1, 1],
                             activation_fn=None,
                             use_batch_norm= False,
                             use_bias= True
                             ) (net, is_training=is_training)
            if self._spatial_squeeze:
                logits = tf.squeeze (logits, (1, 2), name='SpatialSqueeze')
            endpoints['logits'] = logits

        return logits , endpoints

if __name__ == "__main__":
    try:
        from model import  resnet, inception_v1, inception_v2, vgg , SE, densenet
    except:
        import resnet, inception_v1, inception_v2, vgg , SE, densenet
    # tf.enable_eager_execution()
    rgb_input = tf.placeholder (tf.float32,
                                            [None, 224, 224, 3])
        
    flow_input = tf.placeholder (tf.float32,
                                        [None, 224, 224, 20])

    rgb_logits, rgb_endpoints = resnet.Resnet() (rgb_input, is_training=False,
                                                    dropout_keep_prob=1.0)

    flow_logits, flow_endpoints = resnet.Resnet() (flow_input, is_training=False,
                                                dropout_keep_prob=1.0)

    feature_list = [rgb_endpoints,flow_endpoints]
    with tf.variable_scope('fusion'):
        fusion_logits, endpoints = space_cross_correlation_Network() (feature_list, is_training=True, dropout_keep_prob=1.0)
    
    model_logits = fusion_logits
    print(model_logits)
    
    for i in tf.global_variables('fusion'):
        print(i)