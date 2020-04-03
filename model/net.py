from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
sys.path.append('..')
import sonnet as snt
import tensorflow as tf
from model.SMART import Unit3D,Unit_Xseparate3D,Unit_Xseparate3D_v2,mixed_Unit2D,Bottleneck
from tensorflow.contrib.slim.nets import vgg


class ResNet18_3D(snt.AbstractModule):

    VALID_ENDPOINTS = (
        'conv1',
        'conv2_x',
        'conv3_x',
        'conv4_x',
        'conv5_x',
        'Logits',
        'Predictions',
    )
    def __init__(self,num_classes = 101,spatia_squeeze = True,
                 final_endpoint = 'Logits',name = 'ResNet18_3d'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(ResNet18_3D, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint

    def _build(self, inputs,is_training,dropout_prob = 0.5,conv_fn = Unit3D):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('the final end point is not in VALID ENDPOINTS')

        net = inputs
        end_points = {}
        end_point = 'conv1'
        net = conv_fn(output_channels=32,kernel_shape=(3,7,7),stride=(2,2,2))(net,is_training=is_training)

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        end_point = 'conv2_x'
        shortcut = conv_fn(output_channels=64,activation_fn=None,use_batch_norm=False)(net,is_training=is_training)
        net = conv_fn (output_channels=64, kernel_shape=(3, 3, 3), stride=(1, 1, 1)) (net, is_training=is_training)
        net = conv_fn (output_channels=64, kernel_shape=(3, 3, 3), stride=(1, 1, 1)) (net, is_training=is_training)
        net = shortcut + net

        shortcut = tf.nn.max_pool3d (net, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], 'SAME')
        net = conv_fn (output_channels=64, kernel_shape=(3, 3, 3), stride=(1, 1, 1)) (net, is_training=is_training)
        net = conv_fn (output_channels=64, kernel_shape=(3, 3, 3), stride=(2, 2, 2)) (net, is_training=is_training)
        net = shortcut + net
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv3_x'
        shortcut = conv_fn(output_channels=128,activation_fn=None,use_batch_norm=False)(net,is_training=is_training)
        net = conv_fn (output_channels=128, kernel_shape=(3, 3, 3), stride=(1, 1, 1)) (net, is_training=is_training)
        net = conv_fn (output_channels=128, kernel_shape=(3, 3, 3), stride=(1, 1, 1)) (net, is_training=is_training)
        net = shortcut + net

        shortcut = tf.nn.max_pool3d(net,[1,3,3,3,1],[1,1,2,2,1],'SAME')
        net = conv_fn (output_channels=128, kernel_shape=(3, 3, 3), stride=(1, 1, 1)) (net, is_training=is_training)
        net = conv_fn (output_channels=128, kernel_shape=(3, 3, 3), stride=(1, 2, 2)) (net, is_training=is_training)
        net = shortcut + net
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv4_x'
        shortcut = conv_fn(output_channels=256,activation_fn=None,use_batch_norm=False)(net,is_training=is_training)
        net = conv_fn (output_channels=256, kernel_shape=(3, 3, 3), stride=(1, 1, 1)) (net, is_training=is_training)
        net = conv_fn (output_channels=256, kernel_shape=(3, 3, 3), stride=(1, 1, 1)) (net, is_training=is_training)
        net = shortcut + net

        shortcut = tf.nn.max_pool3d (net, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], 'SAME')
        net = conv_fn (output_channels=256, kernel_shape=(3, 3, 3), stride=(1, 1, 1)) (net, is_training=is_training)
        net = conv_fn (output_channels=256, kernel_shape=(3, 3, 3), stride=(2, 2, 2)) (net, is_training=is_training)
        net = shortcut + net
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv5_x'
        shortcut = conv_fn(output_channels=512,activation_fn=None,use_batch_norm=False)(net,is_training=is_training)
        net = conv_fn (output_channels=512, kernel_shape=(3, 3, 3), stride=(1, 1, 1)) (net, is_training=is_training)
        net = conv_fn (output_channels=512, kernel_shape=(3, 3, 3), stride=(1, 1, 1)) (net, is_training=is_training)
        net = shortcut + net

        shortcut = tf.nn.max_pool3d (net, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], 'SAME')
        net = conv_fn (output_channels=512, kernel_shape=(3, 3, 3), stride=(1, 1, 1)) (net, is_training=is_training)
        net = conv_fn (output_channels=512, kernel_shape=(3, 3, 3), stride=(2, 2, 2)) (net, is_training=is_training)
        net = shortcut + net
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points



        end_point = 'Logits'
        net = tf.nn.avg_pool3d(net,ksize=[1,1,7,7,1],strides=[1,1,1,1,1],padding='VALID')
        net = tf.nn.dropout(net,keep_prob=dropout_prob)
        logit = conv_fn(output_channels=self._num_classes,
                        kernel_shape=[1,1,1],
                        activation_fn=None,
                        use_batch_norm=False,
                        use_bias=True)(net,is_training=is_training)
        if self._spatia_squeeze:
            logit = tf.squeeze(logit,(2,3))
        averge_logits = tf.reduce_mean(logit,axis=1)
        end_points[end_point] = averge_logits
        if self._final_endpoint == end_point: return averge_logits, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax (averge_logits)
        end_points[end_point] = predictions
        return predictions, end_points

class ResNet_2D(snt.AbstractModule):

    VALID_ENDPOINTS = (
        'conv1',
        'conv2_x',
        'conv3_x',
        'conv4_x',
        'conv5_x',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=101, spatia_squeeze=True,blocks = [3,4,6,3],
                 final_endpoint='Logits', name='ResNet_2d'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError ('Unknown final endpoint %s' % final_endpoint)

        super (ResNet_2D, self).__init__ (name=name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint
        self._blocks = blocks

    def _build(self, inputs, st_feature_map,is_training, dropout_keep_prob=0.5 ):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError ('the final end point is not in VALID ENDPOINTS')
        if inputs is None and st_feature_map is None:
            raise ValueError ('the inputs and spatio-temporal map are both none')
        net = inputs
        end_points = {}
        end_point = 'conv1'
        # injection mixed spatio-temporal feature (112,112,None) ,input feature (None)
        with tf.variable_scope(end_point):
            if st_feature_map is not None and end_point in st_feature_map:
                spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                net = mixed_Unit2D(output_channels=64,kernel_shape=(7,7))(net,spatial,temporal,is_training=is_training)
            net = snt.Conv2D (output_channels=64, kernel_shape=(7, 7), stride=(1, 1)) (net)
            net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding=snt.SAME)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv2_x'
        with tf.variable_scope(end_point):
            if st_feature_map is not None and end_point in st_feature_map:
                spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                net = mixed_Unit2D(output_channels=64,kernel_shape=(3,3))(net,spatial,temporal,is_training=is_training)
            for i in range(self._blocks[0]):
                if i < self._blocks[0] - 1 :
                    net = Bottleneck(256,64,1,name='block1/unit_'+str(i+1))(net,is_training=is_training)
                else:
                    net = Bottleneck(256,64,2,name='block1/unit_'+str(i+1))(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv3_x'
        # injection spatio-temporal feature (56,56,128),input (56,56,256)
        with tf.variable_scope(end_point):
            if st_feature_map is not None and end_point in st_feature_map:
                spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                net = mixed_Unit2D(output_channels=128,kernel_shape=(3,3))(net,spatial,temporal,is_training=is_training)
            for i in range(self._blocks[1]):
                if i < self._blocks[1] - 1 :
                    net = Bottleneck(512,128,1,name='block2/unit_'+str(i+1))(net,is_training=is_training)
                else:
                    net = Bottleneck(512,128,2,name='block2/unit_'+str(i+1))(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv4_x'
        # injection spatio-temporal feature (28,28,128),input (28,28,64)
        with tf.variable_scope(end_point):
            if st_feature_map is not None and end_point in st_feature_map:
                spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                net = mixed_Unit2D(output_channels=256,kernel_shape=(3,3))(net,spatial,temporal,is_training=is_training)
            for i in range(self._blocks[2]):
                if i < self._blocks[2] - 1 :
                    net = Bottleneck(1024,256,1,name='block3/unit_'+str(i+1))(net,is_training=is_training)
                else:
                    net = Bottleneck(1024,256,2,name='block3/unit_'+str(i+1))(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv5_x'
        # injection spatio-temporal feature (14,14,512),input (14,14,256)
        with tf.variable_scope(end_point):
            if st_feature_map is not None and end_point in st_feature_map:
                spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                net = mixed_Unit2D(output_channels=512,kernel_shape=(3,3))(net,spatial,temporal,is_training=is_training)
            for i in range(self._blocks[3]):
                if i < self._blocks[3] - 1 :
                    net = Bottleneck(2048,512,1,name='block4/unit_'+str(i+1))(net,is_training=is_training)
                else:
                    net = Bottleneck(2048,512,1,name='block4/unit_'+str(i+1))(net,is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Logits'
        # injection spatio-temporal feature (7,7,512),input (7,7,512)
        with tf.variable_scope(end_point):
            kernel = net.shape.as_list()[1]
            net = tf.nn.avg_pool (net, ksize=[1, kernel, kernel, 1], strides=[1, 1, 1, 1], padding='VALID')
            net = tf.nn.dropout (net, keep_prob=dropout_keep_prob)
            logit = snt.Conv2D (output_channels=self._num_classes,
                             kernel_shape=[1, 1],
                             use_bias=True) (net )
            if self._spatia_squeeze:
                logit = tf.squeeze (logit, (1, 2))
            # averge_logits = tf.reduce_mean (logit, axis=1)
        end_points[end_point] = logit
        if self._final_endpoint == end_point: return logit, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax (logit)
        end_points[end_point] = predictions
        return predictions, end_points

class ResNet18_2D(snt.AbstractModule):

    VALID_ENDPOINTS = (
        'conv1',
        'conv2_x',
        'conv3_x',
        'conv4_x',
        'conv5_x',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=101, spatia_squeeze=True,
                 final_endpoint='Logits', name='ResNet18_2d'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError ('Unknown final endpoint %s' % final_endpoint)

        super (ResNet18_2D, self).__init__ (name=name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint

    def _build(self, inputs, st_feature_map,is_training, dropout_keep_prob=1.0 ):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError ('the final end point is not in VALID ENDPOINTS')
        if inputs is None and st_feature_map is None:
            raise ValueError ('the inputs and spatio-temporal map are both none')
        net = inputs
        end_points = {}
        end_point = 'conv1'
        # injection mixed spatio-temporal feature (112,112,None) ,input feature (None)]
        # print(st_feature_map)
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                if st_feature_map is not None and end_point in st_feature_map:
                    # print('conv1 come to finish')
                    spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                    net = mixed_Unit2D(output_channels=64,kernel_shape=(1,1))(net,spatial,temporal,is_training=is_training)
                # net = snt.Conv2D (output_channels=64, kernel_shape=(7, 7), stride=(2, 2)) (net)
                net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding=snt.SAME)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points
        # print(net.shape,net.dtype)

        end_point = 'conv2_x'
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                # bn = snt.BatchNorm()
                # net = bn(net,is_training=is_training,test_local_stats=False)
                net = tf.nn.relu(net)
                shortcut = snt.Conv2D(output_channels=128,kernel_shape=[1,1])(net)
                if st_feature_map is not None and end_point in st_feature_map:
                    spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                    net = mixed_Unit2D(output_channels=128,kernel_shape=(1,1))(net,spatial,temporal,is_training=is_training)
                net = shortcut + net
                net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding=snt.SAME)

                # shortcut = snt.Conv2D (output_channels=64, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = snt.Conv2D (output_channels=64, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = snt.Conv2D (output_channels=64, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = shortcut + net
                #
                # shortcut = tf.nn.max_pool (net, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
                # net = snt.Conv2D (output_channels=64, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = snt.Conv2D (output_channels=64, kernel_shape=(3, 3), stride=(2, 2)) (net )
                # net = shortcut + net
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv3_x'
        # injection spatio-temporal feature (56,56,128),input (56,56,64)
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                net = tf.nn.relu (net)
                shortcut = snt.Conv2D (output_channels=256, kernel_shape=[1, 1])(net)
                if st_feature_map is not None and end_point in st_feature_map:
                    spatial, temporal = st_feature_map[end_point][0], st_feature_map[end_point][1]
                    net = mixed_Unit2D (output_channels=256, kernel_shape=(1, 1)) (net, spatial, temporal,
                                                                                   is_training=is_training)
                net = shortcut + net
                net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding=snt.SAME)

                # shortcut = snt.Conv2D (output_channels=128, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = snt.Conv2D (output_channels=128, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = snt.Conv2D (output_channels=128, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = shortcut + net
                #
                # shortcut = tf.nn.max_pool (net, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
                # net = snt.Conv2D (output_channels=128, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = snt.Conv2D (output_channels=128, kernel_shape=(3, 3), stride=(2, 2)) (net )
                # net = shortcut + net
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv4_x'
        # injection spatio-temporal feature (28,28,128),input (28,28,64)
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                net = tf.nn.relu(net)
                shortcut = snt.Conv2D(output_channels=512,kernel_shape=[1,1])(net)
                if st_feature_map is not None and end_point in st_feature_map:
                    spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                    net = mixed_Unit2D(output_channels=512,kernel_shape=(1,1))(net,spatial,temporal,is_training=is_training)
                net = shortcut + net
                net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding=snt.SAME)

                # shortcut = snt.Conv2D (output_channels=256, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = snt.Conv2D (output_channels=256, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = snt.Conv2D (output_channels=256, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = shortcut + net
                #
                # shortcut = tf.nn.max_pool (net, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
                # net = snt.Conv2D (output_channels=256, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = snt.Conv2D (output_channels=256, kernel_shape=(3, 3), stride=(2, 2)) (net)
                # net = shortcut + net
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv5_x'
        # injection spatio-temporal feature (14,14,512),input (14,14,256)
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                net = tf.nn.relu (net)
                shortcut = snt.Conv2D (output_channels=512, kernel_shape=[1, 1])(net)
                if st_feature_map is not None and end_point in st_feature_map:
                    spatial, temporal = st_feature_map[end_point][0], st_feature_map[end_point][1]
                    net = mixed_Unit2D (output_channels=512, kernel_shape=(1, 1)) (net, spatial, temporal,
                                                                                   is_training=is_training)
                net = shortcut + net
                # net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,1,1,1],padding=snt.SAME)

                # shortcut = snt.Conv2D (output_channels=512, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = snt.Conv2D (output_channels=512, kernel_shape=(3, 3), stride=(1, 1)) (net)
                # net = snt.Conv2D (output_channels=512, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = shortcut + net
                #
                # shortcut = tf.nn.max_pool (net, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
                # net = snt.Conv2D (output_channels=512, kernel_shape=(3, 3), stride=(1, 1)) (net )
                # net = snt.Conv2D (output_channels=512, kernel_shape=(3, 3), stride=(2, 2)) (net )
                # net = shortcut + net
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

        end_point = 'Logits'
        # injection spatio-temporal feature (7,7,512),input (7,7,512)
        with tf.variable_scope(end_point):
            kernel = net.shape.as_list()[1]
            net = tf.nn.avg_pool (net, ksize=[1, kernel, kernel, 1], strides=[1, 1, 1, 1], padding='VALID')
            net = tf.nn.dropout (net, keep_prob=dropout_keep_prob)
            logit = snt.Conv2D (output_channels=self._num_classes,
                             kernel_shape=[1, 1],
                             use_bias=True) (net )
            if self._spatia_squeeze:
                logit = tf.squeeze (logit, (1, 2))
            # averge_logits = tf.reduce_mean (logit, axis=1)
        end_points[end_point] = logit
        if self._final_endpoint == end_point: return logit, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax (logit)
        end_points[end_point] = predictions
        return predictions, end_points

class m2d_resnet(snt.AbstractModule):

    VALID_ENDPOINTS = (
        'conv1',
        'conv2_x',
        'conv3_x',
        'conv4_x',
        'conv5_x',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=101, spatia_squeeze=True,
                 final_endpoint='Logits', name='m2d_resnet'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError ('Unknown final endpoint %s' % final_endpoint)

        super (m2d_resnet, self).__init__ (name=name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint

    def _build(self, inputs, st_feature_map,is_training, dropout_keep_prob=1.0 ):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError ('the final end point is not in VALID ENDPOINTS')
        if inputs is None and st_feature_map is None:
            raise ValueError ('the inputs and spatio-temporal map are both none')
        net = inputs
        end_points = {}
        end_point = 'conv1'
        # injection mixed spatio-temporal feature (112,112,None) ,input feature (None)]
        # print(st_feature_map)
        # start from resnet_v1_50/block1 to resnet_v1_50/block4
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                if st_feature_map is not None and end_point in st_feature_map:
                    # print('conv1 come to finish')
                    spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                    net = mixed_Unit2D(output_channels=64,kernel_shape=(1,1))(net,spatial,temporal,is_training=is_training)
                net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding=snt.SAME)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points
        # print(net.shape,net.dtype)

        end_point = 'conv2_x'
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                # bn = snt.BatchNorm()
                # net = bn(net,is_training=is_training,test_local_stats=False)
                net = tf.nn.relu(net)
                shortcut = snt.Conv2D(output_channels=256,kernel_shape=[1,1])(net)
                if st_feature_map is not None and end_point in st_feature_map:
                    spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                    net = mixed_Unit2D(output_channels=256,kernel_shape=(1,1))(net,spatial,temporal,is_training=is_training)
                net = shortcut + net
                net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding=snt.SAME)


            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv3_x'
        # injection spatio-temporal feature (56,56,128),input (56,56,64)
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                net = tf.nn.relu (net)
                shortcut = snt.Conv2D (output_channels=512, kernel_shape=[1, 1])(net)
                if st_feature_map is not None and end_point in st_feature_map:
                    spatial, temporal = st_feature_map[end_point][0], st_feature_map[end_point][1]
                    net = mixed_Unit2D (output_channels=512, kernel_shape=(1, 1)) (net, spatial, temporal,
                                                                                   is_training=is_training)
                net = shortcut + net
                net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding=snt.SAME)


            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv4_x'
        # injection spatio-temporal feature (28,28,128),input (28,28,64)
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                net = tf.nn.relu(net)
                shortcut = snt.Conv2D(output_channels=1024,kernel_shape=[1,1])(net)
                if st_feature_map is not None and end_point in st_feature_map:
                    spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                    net = mixed_Unit2D(output_channels=1024,kernel_shape=(1,1))(net,spatial,temporal,is_training=is_training)
                net = shortcut + net
                net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,1,1,1],padding=snt.SAME)


            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv5_x'
        # injection spatio-temporal feature (14,14,512),input (14,14,256)
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                net = tf.nn.relu (net)
                shortcut = snt.Conv2D (output_channels=2048, kernel_shape=[1, 1])(net)
                if st_feature_map is not None and end_point in st_feature_map:
                    spatial, temporal = st_feature_map[end_point][0], st_feature_map[end_point][1]
                    net = mixed_Unit2D (output_channels=2048, kernel_shape=(1, 1)) (net, spatial, temporal,
                                                                                   is_training=is_training)
                net = shortcut + net

            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

        end_point = 'Logits'
        # injection spatio-temporal feature (7,7,512),input (7,7,512)
        with tf.variable_scope(end_point):
            kernel = net.shape.as_list()[1]
            net = tf.nn.avg_pool (net, ksize=[1, kernel, kernel, 1], strides=[1, 1, 1, 1], padding='VALID')
            net = tf.nn.dropout (net, keep_prob=dropout_keep_prob)
            logit = snt.Conv2D (output_channels=self._num_classes,
                             kernel_shape=[1, 1],
                             use_bias=True) (net )
            if self._spatia_squeeze:
                logit = tf.squeeze (logit, (1, 2))
            # averge_logits = tf.reduce_mean (logit, axis=1)
        end_points[end_point] = logit
        if self._final_endpoint == end_point: return logit, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax (logit)
        end_points[end_point] = predictions
        return predictions, end_points

class m2d_inceptionV1(snt.AbstractModule):

    VALID_ENDPOINTS = (
        'conv1',
        'conv2_x',
        'conv3_x',
        'conv4_x',
        'conv5_x',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=101, spatia_squeeze=True,
                 final_endpoint='Logits', name='m2d_resnet'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError ('Unknown final endpoint %s' % final_endpoint)

        super (m2d_inceptionV1, self).__init__ (name=name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint

    def _build(self, inputs, st_feature_map,is_training, dropout_keep_prob=1.0 ):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError ('the final end point is not in VALID ENDPOINTS')
        if inputs is None and st_feature_map is None:
            raise ValueError ('the inputs and spatio-temporal map are both none')
        net = inputs
        end_points = {}
        end_point = 'conv1'
        # injection mixed spatio-temporal feature (112,112,None) ,input feature (None)]
        # print(st_feature_map)
        # start from resnet_v1_50/block1 to resnet_v1_50/block4
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                if st_feature_map is not None and end_point in st_feature_map:
                    # print('conv1 come to finish')
                    spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                    net = mixed_Unit2D(output_channels=64,kernel_shape=(1,1))(net,spatial,temporal,is_training=is_training)
                # net = snt.Conv2D (output_channels=64, kernel_shape=(7, 7), stride=(2, 2)) (net)
                net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding=snt.SAME)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points
        # print(net.shape,net.dtype)

        end_point = 'conv2_x'
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                # bn = snt.BatchNorm()
                # net = bn(net,is_training=is_training,test_local_stats=False)
                net = tf.nn.relu(net)
                shortcut = snt.Conv2D(output_channels=192,kernel_shape=[1,1])(net)
                if st_feature_map is not None and end_point in st_feature_map:
                    spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                    net = mixed_Unit2D(output_channels=192,kernel_shape=(1,1))(net,spatial,temporal,is_training=is_training)
                net = shortcut + net
                net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding=snt.SAME)


            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv3_x'
        # injection spatio-temporal feature (56,56,128),input (56,56,64)
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                net = tf.nn.relu (net)
                shortcut = snt.Conv2D (output_channels=256, kernel_shape=[1, 1])(net)
                if st_feature_map is not None and end_point in st_feature_map:
                    spatial, temporal = st_feature_map[end_point][0], st_feature_map[end_point][1]
                    net = mixed_Unit2D (output_channels=256, kernel_shape=(1, 1)) (net, spatial, temporal,
                                                                                   is_training=is_training)
                net = shortcut + net
                net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding=snt.SAME)


            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv4_x'
        # injection spatio-temporal feature (28,28,128),input (28,28,64)
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                net = tf.nn.relu(net)
                shortcut = snt.Conv2D(output_channels=512,kernel_shape=[1,1])(net)
                if st_feature_map is not None and end_point in st_feature_map:
                    spatial , temporal = st_feature_map[end_point][0] , st_feature_map[end_point][1]
                    net = mixed_Unit2D(output_channels=512,kernel_shape=(1,1))(net,spatial,temporal,is_training=is_training)
                net = shortcut + net
                net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding=snt.SAME)


            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv5_x'
        # injection spatio-temporal feature (14,14,512),input (14,14,256)
        if end_point in st_feature_map:
            with tf.variable_scope(end_point):
                net = tf.nn.relu (net)
                shortcut = snt.Conv2D (output_channels=1024, kernel_shape=[1, 1])(net)
                if st_feature_map is not None and end_point in st_feature_map:
                    spatial, temporal = st_feature_map[end_point][0], st_feature_map[end_point][1]
                    net = mixed_Unit2D (output_channels=1024, kernel_shape=(1, 1)) (net, spatial, temporal,
                                                                                   is_training=is_training)
                net = shortcut + net

            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

        end_point = 'Logits'
        # injection spatio-temporal feature (7,7,512),input (7,7,512)
        with tf.variable_scope(end_point):
            kernel = net.shape.as_list()[1]
            net = tf.nn.avg_pool (net, ksize=[1, kernel, kernel, 1], strides=[1, 1, 1, 1], padding='VALID')
            net = tf.nn.dropout (net, keep_prob=dropout_keep_prob)
            logit = snt.Conv2D (output_channels=self._num_classes,
                             kernel_shape=[1, 1],
                             use_bias=True) (net )
            if self._spatia_squeeze:
                logit = tf.squeeze (logit, (1, 2))
            # averge_logits = tf.reduce_mean (logit, axis=1)
        end_points[end_point] = logit
        if self._final_endpoint == end_point: return logit, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax (logit)
        end_points[end_point] = predictions
        return predictions, end_points

class vgg16_2D(snt.AbstractModule):

    VALID_ENDPOINTS = {
        'conv1',
        'conv2',
        'conv3',
        'conv4',
        'conv5',
        'fc6',
        'fc7',
        'fc8',
    }
    def __init__(self,num_classes = 1000,spatia_squeeze = True,
                 final_endpoint = 'Logits',name = 'vgg_16'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(vgg16_2D, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint

    def _build(self, inputs , is_training,dropout_keep_prob = 0.5):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('the final end point is not in VALID ENDPOINTS')

        net = inputs
        end_points = {}
        end_point = 'conv1'
        with tf.variable_scope(end_point):
            net = snt.Conv2D(output_channels=64,kernel_shape=(3,3),stride=(1,1))(net)
            net = snt.Conv2D (output_channels=64, kernel_shape=(3, 3), stride=(1,1)) (net)
            net = tf.nn.max_pool(net,ksize=(1,2,2,1),strides=(1,2,2,1))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points

        end_point = 'conv2'
        with tf.variable_scope(end_point):
            net = snt.Conv2D(output_channels=128,kernel_shape=(3,3),stride=(1,1))(net)
            net = snt.Conv2D (output_channels=128, kernel_shape=(3, 3), stride=(1,1)) (net)
            net = tf.nn.max_pool(net,ksize=(1,2,2,1),strides=(1,2,2,1))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv3'
        with tf.variable_scope(end_point):
            net = snt.Conv2D(output_channels=256,kernel_shape=(3,3),stride=(1,1))(net)
            net = snt.Conv2D (output_channels=256, kernel_shape=(3, 3), stride=(1,1)) (net)
            net = snt.Conv2D (output_channels=256, kernel_shape=(3, 3), stride=(1,1)) (net)
            net = tf.nn.max_pool(net,ksize=(1,2,2,1),strides=(1,2,2,1))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv4'
        with tf.variable_scope(end_point):
            net = snt.Conv2D(output_channels=512,kernel_shape=(3,3),stride=(1,1))(net)
            net = snt.Conv2D (output_channels=512, kernel_shape=(3, 3), stride=(1,1)) (net)
            net = snt.Conv2D (output_channels=512, kernel_shape=(3, 3), stride=(1,1)) (net)
            net = tf.nn.max_pool(net,ksize=(1,2,2,1),strides=(1,2,2,1))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv5'
        with tf.variable_scope(end_point):
            net = snt.Conv2D(output_channels=512,kernel_shape=(3,3),stride=(1,1))(net)
            net = snt.Conv2D (output_channels=512, kernel_shape=(3, 3), stride=(1,1)) (net)
            net = snt.Conv2D (output_channels=512, kernel_shape=(3, 3), stride=(1,1)) (net)
            net = tf.nn.max_pool(net,ksize=(1,2,2,1),strides=(1,2,2,1))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'fc6'
        with tf.variable_scope (end_point):
            net = tf.contrib.layers.fully_connected(net,4096)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'fc7'
        with tf.variable_scope (end_point):
            net = tf.contrib.layers.fully_connected(net,4096)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'fc8'
        with tf.variable_scope (end_point):
            net = tf.contrib.layers.fully_connected(net,1000)
            net = tf.nn.dropout(net,keep_prob=dropout_keep_prob)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points


ENDPOINTS = (
        'conv1',
        'conv2_x',
        'conv3_x',
        'conv4_x',
        'conv5_x',
)
rgb_map = (
    'RGB/vgg_16/pool1',
    'RGB/vgg_16/pool2',
    'RGB/vgg_16/pool3',
    'RGB/vgg_16/pool4',
    'RGB/vgg_16/pool5'
)
flow_map = (
    'Flow/vgg_16/pool1',
    'Flow/vgg_16/pool2',
    'Flow/vgg_16/pool3',
    'Flow/vgg_16/pool4',
    'Flow/vgg_16/pool5'
)
def two_stream_mixed_vgg(rgb_input,flow_input,is_training,pretrainend_model = r'data\vgg_16_2016_08_28\vgg_16.ckpt'):

    with tf.variable_scope('RGB',reuse=tf.AUTO_REUSE):
        rgb_logits , rgb_endpoints = vgg.vgg_16(rgb_input,is_training=is_training)
        for var in tf.global_variables():
            if var.name.split('/') == 'RGB':
                var = tf.contrib.framework.load_variable(pretrainend_model,var.name[4:])

    with tf.variable_scope('Flow',reuse=tf.AUTO_REUSE):
        flow_logits , flow_endpoints = vgg.vgg_16(flow_input,is_training=is_training)
        for var in tf.global_variables():
            if var.name.split('/') == 'Flow':
                var = tf.contrib.framework.load_variable(pretrainend_model,var.name[5:])

    var_map = {}
    for i,endpoint in enumerate(ENDPOINTS):
        var_map[endpoint] = [rgb_endpoints[rgb_map[i]],flow_endpoints[flow_map[i]]]
    print(var_map)
    with tf.variable_scope('mixed',reuse=tf.AUTO_REUSE):
        logits , endpoints = ResNet18_2D()(None,var_map,is_training=True,dropout_keep_prob = 0.5)
    return rgb_endpoints,flow_endpoints

if __name__ == '__main__':
    # pass
    rgb_input = tf.placeholder(tf.float32,[None,112,112,3])
    ResNet_2D()(rgb_input,st_feature_map = None,is_training=True)
    for i in tf.global_variables():
        print(i)
    # flow_input = tf.placeholder(tf.float32,[None,224,224,3])
    # r,f = two_stream_mixed_vgg(rgb_input,flow_input,True)
    # for i in r:
    #     print(i,r[i])
    # for j in f:
    #     print(j,f[j])
    # for i in rgb_map:
    #     print(r[i])
    # print(tf.get_variable(rgb_map[1]))