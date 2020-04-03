import tensorflow as tf
import sonnet as snt


initializer = {
    'w':tf.contrib.layers.xavier_initializer()
}
regularizers = {
    'w':tf.contrib.layers.l2_regularizer(scale=1e-4)
}

sample = tf.image.resize_images

class Unit2D(snt.AbstractModule):

    def __init__(self,output_channels,
                 kernel_shape = (1,1),
                 stride = (1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 use_diag_init = False,
                 name = 'unit_2d'):
        super(Unit2D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._use_diag_init = use_diag_init

    def _build(self, inputs , is_training):
        if self._use_diag_init:
            import numpy as np
            depth = inputs.shape.as_list()[-1]
            init = np.concatenate([np.identity(depth // 2)] * 2)
            initializer = {'w':tf.constant_initializer(init)}
        else:
            initializer = {
                'w': tf.contrib.layers.xavier_initializer ()
            }
        net = snt.Conv2D(output_channels=self._output_channels,
                         kernel_shape=self._kernel_shape,
                         stride=self._stride,
                         padding=snt.SAME,initializers=initializer,
                         use_bias=self._use_bias,regularizers=regularizers)(inputs)
        if self._use_batch_norm:
            bn = snt.BatchNormV2(scale=True)
            net = bn(net,is_training=is_training,test_local_stats = False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net

class FeaturePyramidNetwork(snt.AbstractModule):

    def __init__(self,num_classes = 101,spatial_squeeze=True,name = 'FPN'):
        super(FeaturePyramidNetwork, self).__init__(name=name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze

    def _build(self, feature_pyramid,is_training=True,dropout_keep_prob=1.0):
        '''

        :param feature_pyramid: dict ,(name , tensor)  or list (tensor)
            as for resnet and vgg_16 , tensor begin with size (56,56)->(28,28)->(14,14)->(7,7)
            as for vgg_16 , tensor can also begin with size (112,112)->(56,56)->(28,28)->(14,14)->(7,7)
        :param is_training: bool , batch_norm parameter
        :param dropout_keep_prob: float32, (0,1] dropout parameter
        :return:a tensor list with same dimension but diffenrent size
        '''
        feature_list = []
        if type(feature_pyramid) == dict:
            for _ , v in feature_pyramid.items():
                feature_list.append(v)
        elif type(feature_pyramid) == list:
            feature_list = feature_pyramid

        endpoints = {}
        tensor_list = []
        feature_nums = len(feature_list)
        for i in range(len(feature_list)):
            if i == 0 :
                with tf.variable_scope('top_%d' % feature_nums - i):
                    top_net = Unit2D(output_channels=256,kernel_shape=(1,1),activation_fn=tf.nn.relu,
                                       use_batch_norm=False)(feature_list[feature_nums - i])
                    endpoints['top_%d' % i] = top_net
                    # tensor_list.append(top_net)
                    height = top_net.shape.as_list()[1]
                    width = top_net.shape.as_list()[2]
                    output = tf.nn.avg_pool(top_net,[1,height,width,1],strides=[1,1,1,1],padding='VALID')
                    tensor_list.append(output)
                    top_net = sample(top_net,(height*2,width*2))


            else:
                with tf.variable_scope('top_%d' % feature_nums - i):
                    lateral_net = Unit2D(output_channels=256,kernel_shape=(1,1),activation_fn=None,
                                       use_batch_norm=None)(feature_list[feature_nums - i])
                    top_net = top_net + lateral_net
                    top_net = Unit2D(output_channels=256,kernel_shape=(3,3),
                                     activation_fn=tf.nn.relu,use_batch_norm=False)(top_net)
                    endpoints['top_%d' % i] = top_net
                    # tensor_list.append(top_net)
                    height = top_net.shape.as_list()[1]
                    width = top_net.shape.as_list()[2]
                    output = tf.nn.avg_pool (top_net, [1, height, width, 1], strides=[1, 1, 1, 1], padding='VALID')
                    tensor_list.append (output)
                    top_net = sample(top_net,(height*2,width*2))

        with tf.variable_scope('logits'):
            net = tf.concat(tensor_list,axis=-1)
            net = tf.nn.dropout(net,keep_prob=dropout_keep_prob)
            logits = Unit2D (output_channels=self._num_classes,
                             kernel_shape=[1, 1],
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits') (net, is_training=is_training)
            if self._spatial_squeeze:
                logits = tf.squeeze (logits, (1, 2), name='SpatialSqueeze')
            endpoints['logits'] = logits


        return  logits,endpoints

if __name__ == '__main__':

    logits_unit = Unit2D (output_channels=101,
                             kernel_shape=[1, 1],
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    sample = [tf.placeholder(tf.float32,[None,1,1,1001])]*3
    for i in range(3):
        logit = logits_unit(sample[i],is_training=True)
    for i in tf.global_variables():
        print(i)