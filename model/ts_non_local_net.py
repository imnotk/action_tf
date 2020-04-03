from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import sonnet as snt

try:
    from ts_fusion_net import Unit3d, bottleneck, deep_bottleneck
except:
    from model.ts_fusion_net import Unit3d, bottleneck, deep_bottleneck

try:
    from ts_non_local_helper import spacetime_nonlocal_fusion , _build_feature
except:
    from model.ts_non_local_helper import spacetime_nonlocal_fusion , _build_feature

initializer = {
    'w':tf.contrib.layers.xavier_initializer()
}
regularizers = {
    'w':tf.contrib.layers.l2_regularizer(scale=5e-4)
}

class FeatureHierachyNetwork(snt.AbstractModule):

    def __init__(self,num_classes=101,middle_channels=256,spatial_squeeze=True,name='FHN',use_batch_norm=False,batch_size = None,fusion_mode = 'add',phase='train'):
        super (FeatureHierachyNetwork, self).__init__ (name=name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._use_batch_norm = use_batch_norm
        self._middle_channels = middle_channels    
        self._batch_size = batch_size
        self._fusion_mode = fusion_mode
        self._phase = phase

    def _build(self, inputs,is_training=False,dropout_keep_prob=1.0):

     
        # we only need feature size of 56,28,14,7

        # inputs include feature size of 56,28,14,7 etc
        
        feature_list = _build_feature(inputs)

        feature_list = feature_list[1:]

        endpoints = {}
        feature_nums = len(feature_list)
        logits_list = []
        for i in range(feature_nums):
            if i == 0 :
                with tf.variable_scope('top_%d' % i):
                    feature = feature_list[i]
                    feature = Unit3d (output_channels=self._middle_channels,
                                            activation_fn=tf.nn.relu,
                                            use_batch_norm=False,
                                            ) (feature, is_training=is_training)

                    mixed_feature = spacetime_nonlocal_fusion(output_channels=self._middle_channels,use_conv=True,use_batch_norm=False,activation_fn=None)(feature,is_training=is_training)
                    
                    mixed_feature = tf.nn.relu(mixed_feature)

                    mixed_feature = deep_bottleneck(depth=self._middle_channels,depth_bottleneck=self._middle_channels // 4,stride=(1,1,1),use_batch_norm=self._use_batch_norm,name='adaption_layer_1')(mixed_feature,is_training=is_training)
                    mixed_feature = deep_bottleneck(depth=self._middle_channels,depth_bottleneck=self._middle_channels // 4,stride=(1,2,2),use_batch_norm=self._use_batch_norm,name='adaption_layer_2')(mixed_feature,is_training=is_training)

                    top_net = mixed_feature
                    print('use intermediate supervision training')
                    with tf.variable_scope('logits'):
                        height = top_net.shape.as_list()[2]
                        width = top_net.shape.as_list()[3]
                        middle_net = tf.nn.avg_pool3d(top_net,[1,1,height,width,1],[1,1,1,1,1],padding=snt.VALID)
                        middle_net = tf.contrib.layers.dropout(middle_net,keep_prob=dropout_keep_prob,is_training=is_training)
                        middle_logits = Unit3d (output_channels=self._num_classes,
                                        kernel_shape=[1, 1, 1],
                                        activation_fn=None,
                                        use_batch_norm=False,
                                        use_bias=True
                                        ) (middle_net, is_training=is_training)
                        if self._spatial_squeeze:
                            middle_logits = tf.squeeze (middle_logits, (2, 3), name='SpatialSqueeze')
                        middle_logits = tf.reduce_mean(middle_logits,axis=1)
                    logits_list.append(middle_logits)

                    endpoints['top_%d' % i] = top_net
            else:
                with tf.variable_scope('top_%d' % i):
                    feature = feature_list[i]
                    feature = Unit3d (output_channels=self._middle_channels,
                                            activation_fn=tf.nn.relu,
                                            use_batch_norm=False,
                                            ) (feature, is_training=is_training)

                    mixed_feature = spacetime_nonlocal_fusion(output_channels=self._middle_channels,use_conv=True,use_batch_norm=False,activation_fn=None)(feature,is_training=is_training)

                    if self._fusion_mode == 'concat':
                        mixed_two_feature = tf.nn.relu(tf.concat([mixed_feature,top_net],axis=-1))
                    elif self._fusion_mode == 'add':
                        mixed_two_feature = tf.nn.relu(mixed_feature + top_net)

                    if i != feature_nums - 1:
                        mixed_two_feature = deep_bottleneck(depth=self._middle_channels,depth_bottleneck=self._middle_channels // 4,stride=(1,1,1),use_batch_norm=self._use_batch_norm,name='adaption_layer_1')(mixed_two_feature,is_training=is_training)
                        top_net = deep_bottleneck(depth=self._middle_channels,depth_bottleneck=self._middle_channels // 4,stride=(1,2,2),use_batch_norm=self._use_batch_norm,name='adaption_layer_2')(mixed_two_feature,is_training=is_training)
                    else:
                        mixed_two_feature = deep_bottleneck(depth=self._middle_channels,depth_bottleneck=self._middle_channels // 4,stride=(1,1,1),use_batch_norm=self._use_batch_norm,name='adaption_layer_1')(mixed_two_feature,is_training=is_training)
                        top_net = deep_bottleneck(depth=self._middle_channels,depth_bottleneck=self._middle_channels // 4,stride=(1,1,1),use_batch_norm=self._use_batch_norm,name='adaption_layer_2')(mixed_two_feature,is_training=is_training)

                    
                    with tf.variable_scope('logits'):
                        height = top_net.shape.as_list()[2]
                        width = top_net.shape.as_list()[3]
                        middle_net = tf.nn.avg_pool3d(top_net,[1,1,height,width,1],[1,1,1,1,1],padding=snt.VALID)
                        middle_net = tf.contrib.layers.dropout(middle_net,keep_prob=dropout_keep_prob,is_training=is_training)
                        middle_logits = Unit3d (output_channels=self._num_classes,
                                        kernel_shape=[1, 1, 1],
                                        activation_fn=None,
                                        use_batch_norm=False,
                                        use_bias=True
                                        ) (middle_net, is_training=is_training)
                        if self._spatial_squeeze:
                            middle_logits = tf.squeeze (middle_logits, (2, 3), name='SpatialSqueeze')
                        middle_logits = tf.reduce_mean(middle_logits,axis=1)
                    logits_list.append(middle_logits)

                    endpoints['top_%d' % i] = top_net


        if self._phase == 'test':
            # with tf.variable_scope('logits'):
            #     net = tf.nn.avg_pool3d(top_net,[1,1,7,7,1],[1,1,1,1,1],padding=snt.VALID)
            #     net = tf.contrib.layers.dropout(net,keep_prob=dropout_keep_prob,is_training=is_training)
            #     logits = Unit3d (output_channels=self._num_classes,
            #                     kernel_shape=[1, 1, 1],
            #                     activation_fn=None,
            #                     use_batch_norm=False,
            #                     use_bias=True
            #                     ) (net, is_training=is_training)
            #     if self._spatial_squeeze:
            #         logits = tf.squeeze (logits, (2, 3), name='SpatialSqueeze')
            #     logits = tf.reduce_mean(logits,axis=1)
            logits = logits_list[-1]
            endpoints['logits'] = logits
            return logits , endpoints
        elif self._phase == 'train':
            logits = tf.reduce_mean(logits_list,axis=0)
            endpoints['logits'] = logits
            return logits,endpoints

if __name__ == "__main__":
    try:
        from model import  TS_resnet
    except:
        import TS_resnet
    # tf.enable_eager_execution()
    rgb_input = tf.placeholder (tf.float32,
                                            [None,3, 224, 224, 3])
        
    flow_input = tf.placeholder (tf.float32,
                                        [None,3, 224, 224, 20])

    rgb_logits, rgb_endpoints = TS_resnet.Resnet() (rgb_input, is_training=False,
                                                    dropout_keep_prob=1.0)

    flow_logits, flow_endpoints = TS_resnet.Resnet() (flow_input, is_training=False,
                                                dropout_keep_prob=1.0)

    with tf.variable_scope('fusion'):
        fusion_logits, endpoints = FeatureHierachyNetwork(phase='train') (rgb_endpoints, is_training=True, dropout_keep_prob=1.0)
    
    model_logits = fusion_logits
    print(fusion_logits)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.losses.get_regularization_loss('fusion')))
    saver = tf.train.Saver(var_list=tf.global_variables('fusion'))
    saver.save(sess,'./test/model.ckpt')
    for i in tf.global_variables('fusion'):
        print(i)