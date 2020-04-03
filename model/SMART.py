from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sonnet as snt
import tensorflow as tf
slim = tf.contrib.slim
initializers = {
    'w':tf.contrib.layers.xavier_initializer(),
    'b':tf.contrib.layers.xavier_initializer()
}

def subsample(inputs,factor,scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)

class Unit3D(snt.AbstractModule):

    def __init__(self,output_channels,
                 kernel_shape = (1,1,1),
                 stride = (1,1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 initializer = None,
                 name = 'unit_3d'):
        super(Unit3D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._initalizer = initializer

    def _build(self, inputs , is_training):
        net = snt.Conv3D(output_channels=self._output_channels,
                         kernel_shape=self._kernel_shape,
                         stride=self._stride,
                         padding=snt.SAME,
                         use_bias=self._use_bias,
                         initializers=self._initalizer)(inputs)
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net,is_training = is_training,test_local_stats = False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net

class Unit_separate3D(snt.AbstractModule):

    def __init__(self,output_channels,
                 kernel_shape = (1,1,1),
                 stride = (1,1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 name = 'unit_separate3d'):
        super(Unit_separate3D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._temporal_shape , self._x_shape,self._y_shape = kernel_shape
        self._temporal_stride , self._x_stride , self._y_stride = stride

    def _build(self,inputs,is_training,is_spatial_first=True,is_separate_pooling=False):
        if is_spatial_first:
            net = snt.Conv3D(output_channels=self._output_channels,
                             kernel_shape=(1,self._x_shape,self._y_shape),
                             stride=(1,self._x_stride,self._y_stride),
                             padding=snt.SAME,
                             use_bias=self._use_bias)(inputs)
            if is_separate_pooling:
                net = tf.nn.max_pool3d(net,ksize=(1,1,3,3,1),strides=(1,1,2,2,1),padding='SAME')
            net = snt.Conv3D(output_channels=self._output_channels,
                             kernel_shape=(self._temporal_shape,1,1),
                             stride=(self._temporal_stride,1,1),
                             padding=snt.SAME,
                             use_bias=self._use_bias)(inputs)
        else:
            net = snt.Conv3D (output_channels=self._output_channels,
                              kernel_shape=(self._temporal_shape, 1, 1),
                              stride=(self._temporal_stride, 1, 1),
                              padding=snt.SAME,
                              use_bias=self._use_bias) (inputs)
            if is_separate_pooling:
                net = tf.nn.max_pool3d (net, ksize=(1, 1, 3, 3, 1), strides=(1, 1, 2, 2, 1), padding='SAME')
            net = snt.Conv3D (output_channels=self._output_channels,
                              kernel_shape=(1, self._x_shape, self._y_shape),
                              stride=(1, self._x_stride, self._y_stride),
                              padding=snt.SAME,
                              use_bias=self._use_bias) (inputs)
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net,is_training=is_training,test_local_stats = False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net

class Unit_Xseparate3D(snt.AbstractModule):

    def __init__(self,output_channels,
                 kernel_shape = (1,1,1),
                 stride = (1,1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 name = 'unit_Xseparate3d'):
        super(Unit_Xseparate3D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._temporal_shape , self._x_shape,self._y_shape = kernel_shape
        self._temporal_stride , self._x_stride , self._y_stride = stride

    def _build(self,inputs,is_training,is_spatial_first=True,is_separate_pooling=False):
        if is_spatial_first:
            net = snt.Conv3D(output_channels=self._output_channels,
                             kernel_shape=(1,self._x_shape,1),
                             stride=(1,self._x_stride,1),
                             padding=snt.SAME,
                             use_bias=self._use_bias)(inputs)
            net = snt.Conv3D (output_channels=self._output_channels,
                              kernel_shape=(1, 1, self._y_stride),
                              stride=(1, 1, self._y_stride),
                              padding=snt.SAME,
                              use_bias=self._use_bias) (net)
            if is_separate_pooling:
                net = tf.nn.max_pool3d(net,ksize=(1,1,3,3,1),strides=(1,1,2,2,1),padding='SAME')
            net = snt.Conv3D(output_channels=self._output_channels,
                             kernel_shape=(self._temporal_shape,1,1),
                             stride=(self._temporal_stride,1,1),
                             padding=snt.SAME,
                             use_bias=self._use_bias)(inputs)
        else:
            net = snt.Conv3D (output_channels=self._output_channels,
                              kernel_shape=(self._temporal_shape, 1, 1),
                              stride=(self._temporal_stride, 1, 1),
                              padding=snt.SAME,
                              use_bias=self._use_bias) (inputs)
            if is_separate_pooling:
                net = tf.nn.max_pool3d (net, ksize=(1, 1, 3, 3, 1), strides=(1, 1, 2, 2, 1), padding='SAME')
            net = snt.Conv3D (output_channels=self._output_channels,
                              kernel_shape=(1, self._x_shape, 1),
                              stride=(1, self._x_stride, 1),
                              padding=snt.SAME,
                              use_bias=self._use_bias) (inputs)
            net = snt.Conv3D (output_channels=self._output_channels,
                              kernel_shape=(1, 1, self._y_stride),
                              stride=(1, 1, self._y_stride),
                              padding=snt.SAME,
                              use_bias=self._use_bias) (net)
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net,is_training=is_training,test_local_stats = False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net

class Unit_Xseparate3D_v2(snt.AbstractModule):

    def __init__(self,output_channels,
                 kernel_shape = (1,1,1),
                 stride = (1,1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 name = 'unit_Xseparate3d'):
        super(Unit_Xseparate3D_v2, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._temporal_shape , self._x_shape,self._y_shape = kernel_shape
        self._temporal_stride , self._x_stride , self._y_stride = stride

    def _build(self,inputs,is_training,is_spatial_first=True,is_separate_pooling=False):
        if is_spatial_first:
            net_x = snt.Conv3D(output_channels=self._output_channels,
                             kernel_shape=(1,self._x_shape,1),
                             stride=(1,self._x_stride,1),
                             padding=snt.SAME,
                             use_bias=self._use_bias)(inputs)
            net_y = snt.Conv3D (output_channels=self._output_channels,
                              kernel_shape=(1, 1, self._y_stride),
                              stride=(1, 1, self._y_stride),
                              padding=snt.SAME,
                              use_bias=self._use_bias) (inputs)
            net = net_x + net_y
            if is_separate_pooling:
                net = tf.nn.max_pool3d(net,ksize=(1,1,3,3,1),strides=(1,1,2,2,1),padding='SAME')
            net = snt.Conv3D(output_channels=self._output_channels,
                             kernel_shape=(self._temporal_shape,1,1),
                             stride=(self._temporal_stride,1,1),
                             padding=snt.SAME,
                             use_bias=self._use_bias)(inputs)
        else:
            net = snt.Conv3D (output_channels=self._output_channels,
                              kernel_shape=(self._temporal_shape, 1, 1),
                              stride=(self._temporal_stride, 1, 1),
                              padding=snt.SAME,
                              use_bias=self._use_bias) (inputs)
            if is_separate_pooling:
                net = tf.nn.max_pool3d (net, ksize=(1, 1, 3, 3, 1), strides=(1, 1, 2, 2, 1), padding='SAME')
            net_x = snt.Conv3D (output_channels=self._output_channels,
                                kernel_shape=(1, self._x_shape, 1),
                                stride=(1, self._x_stride, 1),
                                padding=snt.SAME,
                                use_bias=self._use_bias) (net)
            net_y = snt.Conv3D (output_channels=self._output_channels,
                                kernel_shape=(1, 1, self._y_stride),
                                stride=(1, 1, self._y_stride),
                                padding=snt.SAME,
                                use_bias=self._use_bias) (net)
            net = net_x + net_y
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net,is_training=is_training,test_local_stats = False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net


class Unit_Xseparate3D_v3(snt.AbstractModule):

    def __init__(self,output_channels,
                 kernel_shape = (1,1,1),
                 stride = (1,1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 name = 'unit_Xseparate3d'):
        super(Unit_Xseparate3D_v3, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._temporal_shape , self._x_shape,self._y_shape = kernel_shape
        self._temporal_stride , self._x_stride , self._y_stride = stride

    def _build(self,inputs,is_training,is_spatial_first=True,is_separate_pooling=False):
        if is_spatial_first:
            net_x = snt.Conv3D(output_channels=self._output_channels,
                             kernel_shape=(1,self._x_shape,1),
                             stride=(1,self._x_stride,1),
                             padding=snt.SAME,
                             use_bias=self._use_bias)(inputs)
            net_y = snt.Conv3D (output_channels=self._output_channels,
                              kernel_shape=(1, 1, self._y_stride),
                              stride=(1, 1, self._y_stride),
                              padding=snt.SAME,
                              use_bias=self._use_bias) (inputs)
            net = tf.sqrt(tf.square(net_x)+tf.square(net_y))
            if is_separate_pooling:
                net = tf.nn.max_pool3d(net,ksize=(1,1,3,3,1),strides=(1,1,2,2,1),padding='SAME')
            net = snt.Conv3D(output_channels=self._output_channels,
                             kernel_shape=(self._temporal_shape,1,1),
                             stride=(self._temporal_stride,1,1),
                             padding=snt.SAME,
                             use_bias=self._use_bias)(inputs)
        else:
            net = snt.Conv3D (output_channels=self._output_channels,
                              kernel_shape=(self._temporal_shape, 1, 1),
                              stride=(self._temporal_stride, 1, 1),
                              padding=snt.SAME,
                              use_bias=self._use_bias) (inputs)
            if is_separate_pooling:
                net = tf.nn.max_pool3d (net, ksize=(1, 1, 3, 3, 1), strides=(1, 1, 2, 2, 1), padding='SAME')
            net_x = snt.Conv3D (output_channels=self._output_channels,
                                kernel_shape=(1, self._x_shape, 1),
                                stride=(1, self._x_stride, 1),
                                padding=snt.SAME,
                                use_bias=self._use_bias) (net)
            net_y = snt.Conv3D (output_channels=self._output_channels,
                                kernel_shape=(1, 1, self._y_stride),
                                stride=(1, 1, self._y_stride),
                                padding=snt.SAME,
                                use_bias=self._use_bias) (net)
            net = tf.sqrt(tf.square(net_x)+tf.square(net_y))
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net,is_training=is_training,test_local_stats = False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net

class Unit2D(snt.AbstractModule):

    def __init__(self,output_channels,
                 kernel_shape = (1,1),
                 stride = (1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 name = 'unit_2d'):
        super(Unit2D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        # self._temporal_shape , self._x_shape,self._y_shape = kernel_shape
        # self._temporal_stride , self._x_stride , self._y_stride = stride

    def _build(self, inputs , is_training):
        net = snt.Conv2D(output_channels=self._output_channels,
                         kernel_shape=self._kernel_shape,
                         stride=self._stride,
                         padding=snt.SAME,
                         use_bias=self._use_bias)(inputs)
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net,is_training=is_training,test_local_stats = False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net

class Bottleneck(snt.AbstractModule):

    def __init__(self,depth,depth_bottleneck,
                 stride=(1,1),name='block'):
        super(Bottleneck,self).__init__(name=name)
        self._depth = depth
        self._depth_bottleneck = depth_bottleneck
        self._stride = stride

    def _build(self, inputs , is_training):
        depth_in = inputs.shape.as_list()[-1]
        preact = snt.BatchNorm()(inputs,is_training=is_training,test_local_stats=False)
        preact = tf.nn.relu(preact)

        if depth_in == self._depth:
            shortcut = subsample(preact,self._stride,'shortcut')
        else:
            shortcut = snt.Conv2D(output_channels=self._depth,kernel_shape=[1,1],stride=self._stride,
                                  name='shortcut')(preact)

        residual = snt.Conv2D(self._depth_bottleneck,kernel_shape=[1,1],stride=1,name='conv1')(preact)
        residual = snt.Conv2D(self._depth_bottleneck,kernel_shape=[3,3],stride=self._stride,name='conv2')(residual)
        residual = snt.Conv2D(self._depth,kernel_shape=[1,1],stride=1,name='conv3')(residual)

        output = shortcut + residual
        return output

class mixed_Unit2D(snt.AbstractModule):

    def __init__(self,output_channels,
                 kernel_shape = (1,1),
                 stride = (1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 name = 'mixed_Unit2D'):
        super(mixed_Unit2D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        # self._temporal_shape , self._x_shape,self._y_shape = kernel_shape
        # self._temporal_stride , self._x_stride , self._y_stride = stride

    def _concat(self,spatial_cnn,motion_cnn,
                is_training,
                use_batch_norm=False,
                activation_fn=None):
        # if spatial_cnn.shape[-1] != motion_cnn.shape:
        #     raise ValueError ('spatial cnn shape is ',spatial_cnn.shape ,'not equal to motion cnn shape' ,motion_cnn.shape)
        net = tf.concat([spatial_cnn,motion_cnn],axis = -1)
        if use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net,is_training=is_training,test_local_stats = False)
        if activation_fn is not None:
            net = activation_fn(net)
        return net

    def _mix(self,origin,
             ST,
             is_training,
             use_batch_norm = True,
             activation_fn = None,
             use_pooling = False):
        # if use_pooling:
        #     origin = tf.nn.max_pool(origin,ksize=[1,3,3,1],strides=[1,1,1,1],padding=snt.SAME)
        #     inputs = tf.nn.max_pool(inputs,ksize=[1,3,3,1],strides=[1,1,1,1],padding=snt.SAME)
        # origin = snt.Conv2D(output_channels=self._output_channels,
        #                     kernel_shape=[1,1])(origin)
        ST = snt.Conv2D(output_channels=self._output_channels,
                            kernel_shape=[1,1])(ST)
        if origin.shape.as_list() != ST.shape.as_list():
            raise ValueError(" origin's shape is " , origin.shape.as_list(),"inputs's shape is",ST.shape.as_list())
        # print(origin.shape,ST.shape)
        net = tf.concat(origin,ST)
        if use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net,is_training=is_training,test_local_stats = False)
        if activation_fn is not None:
            net = activation_fn(net)
        return net

    def _build(self, inputs,spatial_cnn,motion_cnn,is_training):

        if spatial_cnn is not None and motion_cnn is not None:
            # print('spatial cnn and motion cnn is not None')
            ST = self._concat(spatial_cnn,motion_cnn,is_training=is_training,activation_fn=None)
            ST = snt.Conv2D(output_channels=self._output_channels,kernel_shape=[1,1])(ST)
            if inputs is not None:
                # net = self._mix(inputs,ST,is_training=is_training)
                net = tf.concat([inputs,ST],axis=-1)
            else:
                # print('inputs is none')
                net = ST
        else:
            if inputs is not None:
                net = inputs

        net = snt.Conv2D (output_channels=self._output_channels,
                          kernel_shape=self._kernel_shape,
                          stride=self._stride,
                          padding=snt.SAME,
                          use_bias=self._use_bias) (net)

        if self._use_batch_norm:
            bn = snt.BatchNorm ()
            net = bn (net, is_training=is_training, test_local_stats=False)
        if self._activation_fn is not None:
            net = self._activation_fn (net)
        return net

class SMART_block(snt.AbstractModule):

    def __init__(self,output_channels,
                 kernel_shape = (1,1,1),
                 stride = (1,1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 name = 'SMART'):
        super(SMART_block, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._temporal_shape , self._x_shape,self._y_shape = kernel_shape
        self._temporal_stride , self._x_stride , self._y_stride = stride

    def _build(self, inputs, is_training):
        spatial_net = Unit3D(output_channels=self._output_channels,
                             kernel_shape=(1,self._x_shape,self._y_shape),
                             stride=(1,self._x_stride,self._y_stride),
                             name='spatial')(inputs,is_training=is_training)
        temporal_net = Unit3D(output_channels=self._output_channels,
                              kernel_shape=self._kernel_shape,
                              stride=self._stride,activation_fn=None,name='temporal')(inputs,is_training=is_training)
        temporal_net = tf.square(temporal_net)
        temporal_net = Unit3D(output_channels=self._output_channels//2,
                              kernel_shape=self._kernel_shape,use_bias=False,
                              initializer={'w':tf.initializers.constant([0.5])})(temporal_net,is_training=is_training)
        net = tf.concat([temporal_net,spatial_net],axis=-1)
        return net

class Separte_SMART_block(snt.AbstractModule):

    def __init__(self,output_channels,
                 kernel_shape = (1,1,1),
                 stride = (1,1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 name = 'SMART'):
        super(Separte_SMART_block, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._temporal_shape , self._x_shape,self._y_shape = kernel_shape
        self._temporal_stride , self._x_stride , self._y_stride = stride

    def _build(self, inputs, is_training):
        spatial_1 = Unit3D(output_channels=self._output_channels,
                           kernel_shape=(1,self._x_shape,self._y_shape),
                           stride=(1,self._x_stride,self._y_stride),
                           activation_fn=None)
        spatial_2 = Unit3D (output_channels=self._output_channels,
                            kernel_shape=(1, self._x_shape, self._y_shape),
                            stride=(1, self._x_stride, self._y_stride),
                            activation_fn=None)
        spatial_1_net = spatial_1(inputs,is_training=is_training)
        spatial_2_net = spatial_2(inputs,is_training=is_training)
        temporal_1 = Unit3D(output_channels=self._output_channels,
                            kernel_shape=(self._temporal_shape,1,1),
                            stride=(self._temporal_stride,1,1))
        temporal_2 = Unit3D (output_channels=self._output_channels,
                             kernel_shape=(self._temporal_shape, 1, 1),
                             stride=(self._temporal_stride, 1, 1))
        spatial_1_1_net = temporal_1(spatial_1_net,is_training=is_training)
        spatial_1_2_net = temporal_2(spatial_1_net,is_training=is_training)
        spatial_2_1_net = temporal_1(spatial_2_net,is_training=is_training)
        spatial_2_2_net = temporal_2(spatial_2_net,is_training=is_training)
        A = spatial_1_1_net * spatial_2_2_net
        B = spatial_2_1_net *spatial_1_2_net
        return A + B

class non_local_block(snt.AbstractModule):


    def __init__(self,output_channels,
                 kernel_shape = (1,1,1),
                 stride = (1,1,1),
                 activation_fn = tf.nn.relu,
                 use_batch_norm = True,
                 use_bias = False,
                 block_function = 'gaussian',
                 name = 'non_local_block'):
        super(non_local_block, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._temporal_shape , self._x_shape,self._y_shape = kernel_shape
        self._temporal_stride , self._x_stride , self._y_stride = stride
        self._block_function = {'gaussion':self.gaussian,
                                'embedded_gaussian':self.embedded_gaussian,
                                'dot_product':self.Dot_product,
                                'concatenation':self.Concatenation
                                }

    def _build(self, inputs , block_function,softmax = False):
        represent = snt.Conv3D(output_channels=self._output_channels//2,
                              kernel_shape=self._kernel_shape,
                              stride=self._stride,use_bias=self._use_bias)(inputs)
        shape_0,shape_1,shape_2,shape_3,shape_4 = represent.shape
        represent = tf.reshape(represent,[represent.shape[0].value,represent.shape[1].value*represent.shape[2].value*represent.shape[3].value,
                                          represent.shape[-1].value])
        if block_function in self._block_function:
            relation = self._block_function[block_function](inputs)
            factor = relation.shape[-1].value
        if softmax:
            relation = relation  / tf.cast(factor,tf.float32)
        else:
            relation = tf.nn.softmax(relation)
        response = tf.matmul(relation,represent)
        response = tf.reshape(response,[shape_0,shape_1,shape_2,shape_3,-1])
        # print(,self._output_channels)
        # assert inputs.shape[-1].value == self._output_channels
        fg = snt.Conv3D(output_channels=self._output_channels,
                        kernel_shape=[1,1,1])(response)
        if inputs.shape[-1].value == self._output_channels:
            net = inputs + fg
        else:
            net = snt.Conv3D(output_channels=self._output_channels,kernel_shape=[1,1,1])(inputs)
            net = net + fg
        return net

    # @classmethod
    def gaussian(self,inputs):
        inputs = tf.reshape(inputs,[inputs.shape[0].value,-1,inputs.shape[-1].value])
        return tf.matmul(inputs,tf.transpose(inputs,[0,2,1]))

    # @classmethod
    def embedded_gaussian(self,inputs):
        inputs_x = snt.Conv3D(output_channels=self._output_channels  // 2,
                              kernel_shape=self._kernel_shape,
                              stride=self._stride,use_bias=self._use_bias)(inputs)
        inputs_x = tf.reshape(inputs_x,[inputs_x.shape[-0].value,-1,inputs_x.shape[-1].value])
        inputs_y = snt.Conv3D (output_channels=self._output_channels // 2,
                               kernel_shape=self._kernel_shape,
                               stride=self._stride, use_bias=self._use_bias) (inputs)
        inputs_y = tf.reshape(inputs_y,[inputs_y.shape[0].value,-1,inputs_y.shape[-1].value])
        return tf.matmul(inputs_x,tf.transpose(inputs_y,[0,2,1]))

    # @classmethod
    def Dot_product(self,inputs):
        inputs_x = snt.Conv3D (output_channels=self._output_channels // 2,
                               kernel_shape=self._kernel_shape,
                               stride=self._stride, use_bias=self._use_bias) (inputs)
        inputs_x = tf.reshape(inputs_x,[inputs_x.shape[-0].value,-1,inputs_x.shape[-1].value])
        inputs_y = snt.Conv3D (output_channels=self._output_channels // 2,
                               kernel_shape=self._kernel_shape,
                               stride=self._stride, use_bias=self._use_bias) (inputs)
        inputs_y = tf.reshape(inputs_y,[inputs_y.shape[0].value,-1,inputs_y.shape[-1].value])
        return tf.matmul(inputs_x,tf.transpose(inputs_y,[0,2,1]))

    # @classmethod
    def Concatenation(self,inputs):
        # TODO
        inputs_x = snt.Conv3D (output_channels=self._output_channels // 2,
                               kernel_shape=self._kernel_shape,
                               stride=self._stride, use_bias=self._use_bias) (inputs)
        inputs_x = tf.reshape(inputs_x,[inputs_x.shape[-0].value,-1,inputs_x.shape[-1].value])
        inputs_y = snt.Conv3D (output_channels=self._output_channels // 2,
                               kernel_shape=self._kernel_shape,
                               stride=self._stride, use_bias=self._use_bias) (inputs)
        inputs_y = tf.reshape(inputs_y,[inputs_y.shape[0].value,-1,inputs_y.shape[-1].value])
        net = tf.concat([inputs_x,inputs_y],axis = -1)
        pass

if __name__ == '__main__':
    smart = SMART_block(output_channels=10)
    separate_smart = Separte_SMART_block(output_channels=10)
    sample = tf.placeholder(tf.float32,[3,10,16,16,10])
    # smart(sample,is_training = True)
    non_local = non_local_block(output_channels=10)
    # non_local(sample,'embedded_gaussian',True)
    separate_smart(sample,is_training=True)