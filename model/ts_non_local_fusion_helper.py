from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import sonnet as snt

try:
    from ts_fusion_net import Unit3d
except:
    from model.ts_fusion_net import Unit3d

initializer = {
    'w':tf.contrib.layers.xavier_initializer()
}
regularizers = {
    'w':tf.contrib.layers.l2_regularizer(scale=1e-4)
}

def linear_transform(inputs,l):

    w = tf.get_variable('w',shape=[1,1,l,1],initializer = tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b',shape=[1,1,l,1],initializer = tf.initializers.constant(0.2))

    return tf.nn.relu(w * inputs + b)

class space_nonlocal_fusion(snt.AbstractModule):

    def __init__(self,batch_size = None,
                 use_conv = False,
                 use_softmax = False,
                 space = True,
                 name = 'space_nonlocal_fusion'):
        super(space_nonlocal_fusion, self).__init__(name=name)
        self._use_conv = use_conv
        self._batch_size = batch_size
        self._use_softmax = use_softmax
        self._space = space


    def _build(self, main_inputs , attention_inputs , is_training):
        
        # rgb_inputs and flow_inputs already has same shape as (None,frame_counts,56,56,256)
        # 
        
        if self._use_conv:
            main_theta = Unit3d(main_inputs.shape.as_list()[-1],activation_fn=None,use_batch_norm=False,name='main_theta')(main_inputs,is_training=is_training)
            attention_theta = Unit3d(attention_inputs.shape.as_list()[-1],activation_fn=None,use_batch_norm=False,name='attention_theta')(attention_inputs,is_training=is_training)
        else:
            main_theta = main_inputs
            attention_theta = attention_inputs
        
        temporal = main_theta.shape.as_list()[1]
        height = main_theta.shape.as_list()[2]
        width = main_theta.shape.as_list()[3]
        dim_inner = main_theta.shape.as_list()[-1]

        if self._batch_size is None:
            self._batch_size = -1

        if self._space:
            main_theta = tf.reshape(main_theta,[self._batch_size, temporal,height * width,dim_inner],name='main_theta_reshape')
            attention_theta = tf.reshape(attention_theta,[self._batch_size, temporal,attention_theta.shape.as_list()[2] * attention_theta.shape.as_list()[3],dim_inner],name='attention_theta_reshape')
        else:
            main_theta = tf.reshape(main_theta,[self._batch_size, height * width,dim_inner * temporal],name='main_theta_reshape')
            attention_theta = tf.reshape(attention_theta,[self._batch_size, attention_theta.shape.as_list()[2] * attention_theta.shape.as_list()[3],dim_inner * temporal],name='attention_theta_reshape')
  
        main_attention_theta = tf.matmul(main_theta,attention_theta,transpose_a=False,transpose_b=True,name='main_attention_theta_matmul')
        l = main_attention_theta.shape.as_list()[3]
        main_attention_theta = linear_transform(main_attention_theta,l)
        if self._use_softmax:
            p = tf.nn.softmax(main_attention_theta,name='main_attention_theta_softmax')
        else:
            p = main_attention_theta
        
        main_attention = tf.matmul(p,attention_theta)
        main_attention = tf.reshape(main_attention,[self._batch_size,temporal,height,width,dim_inner])

        if self._use_conv:
            main_attention = Unit3d(dim_inner,activation_fn=None,use_batch_norm=False)(main_attention,is_training=is_training)

        main_attention = snt.BatchNormV2(scale=True)(main_attention,is_training=is_training,test_local_stats=False)

        return main_attention

class channel_nonlocal_fusion(snt.AbstractModule):

    def __init__(self,batch_size = None,
                 use_conv = False,
                 use_softmax = True,
                 space = True,
                 name = 'channel_nonlocal_fusion'):
        super(channel_nonlocal_fusion, self).__init__(name=name)
        self._use_conv = use_conv
        self._batch_size = batch_size
        self._use_softmax = use_softmax
        self._space = space

    def _build(self, main_inputs , attention_inputs , is_training):
        
        # rgb_inputs and flow_inputs already has same shape as (None,frame_counts,56,56,256)
        # 

        if self._use_conv:
            main_theta = Unit3d(main_inputs.shape.as_list()[-1],activation_fn=None,use_batch_norm=False)(main_inputs,is_training=is_training)
            attention_theta = Unit3d(attention_inputs.shape.as_list()[-1],activation_fn=None,use_batch_norm=False)(attention_inputs,is_training=is_training)
        else:
            main_theta = main_inputs
            attention_theta = attention_inputs
        
        temporal = main_theta.shape.as_list()[1]
        height = main_theta.shape.as_list()[2]
        width = main_theta.shape.as_list()[3]
        dim_inner = main_theta.shape.as_list()[-1]

        if self._batch_size is None:
            self._batch_size = -1

        if self._space:
            main_theta = tf.reshape(main_theta,[self._batch_size, temporal,height * width,dim_inner],name='main_theta_reshape')
            attention_theta = tf.reshape(attention_theta,[self._batch_size, temporal,attention_theta.shape.as_list()[2] * attention_theta.shape.as_list()[3],dim_inner],name='attention_theta_reshape')
        else:
            main_theta = tf.reshape(main_theta,[self._batch_size, height * width,dim_inner * temporal],name='main_theta_reshape')
            attention_theta = tf.reshape(attention_theta,[self._batch_size, attention_theta.shape.as_list()[2] * attention_theta.shape.as_list()[3],dim_inner * temporal],name='attention_theta_reshape')

        main_attention_theta = tf.matmul(main_theta,attention_theta,transpose_a=False,transpose_b=True,name='main_attention_theta_matmul')

        if self._use_softmax:
            p = tf.nn.softmax(main_attention_theta,name='main_attention_theta_softmax')
        else:
            ones = tf.constant(1,shape=main_attention_theta.shape)
            ones = tf.reduce_sum(ones)

            zeros = tf.constant(0,shape=main_attention_theta.shape)
            denom = tf.add(zeros,ones)

            tf.stop_gradient(denom)

            p = tf.div(main_attention_theta,denom,name='main_attention_theta_dot_product')
        
        main_attention = tf.matmul(p,attention_theta)
        main_attention = tf.reshape(main_attention,[self._batch_size,temporal,height,width,dim_inner])

        if self._use_conv:
            main_attention = Unit3d(dim_inner,activation_fn=None,use_batch_norm=False)(main_attention,is_training=is_training)

        main_attention = snt.BatchNormV2(scale=True)(main_attention,is_training=is_training,test_local_stats=False)

        return main_attention

            
if __name__ == "__main__":
    s1 = tf.placeholder(tf.float32,[None,25,28,28,3])
    s2 = tf.placeholder(tf.float32,[None,25,28,28,3])
    a = space_nonlocal_fusion(256)(s1,s2,True)
    for i in tf.global_variables():
        print(i)