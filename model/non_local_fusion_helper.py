from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import sonnet as snt

try:
    from fusion_net import Unit2D
except:
    from model.fusion_net import Unit2D


class space_nonlocal_fusion(snt.AbstractModule):

    def __init__(self,batch_size = None,
                 use_conv = False,
                 use_softmax = True,
                 name = 'space_nonlocal_fusion'):
        super(space_nonlocal_fusion, self).__init__(name=name)
        self._batch_size = batch_size
        self._use_conv = use_conv
        self._use_softmax = use_softmax

    def _build(self, main_inputs , attention_inputs , is_training):
        
        if self._use_conv:
            main_theta = Unit2D(main_inputs.shape.as_list()[-1],activation_fn=None,use_batch_norm=False,name='main_theta')(main_inputs,is_training=is_training)
            attention_theta = Unit2D(attention_inputs.shape.as_list()[-1],activation_fn=None,use_batch_norm=False,name='attention_theta')(attention_inputs,is_training=is_training)
            g_theta = Unit2D(attention_inputs.shape.as_list()[-1],activation_fn=None,use_batch_norm=False,name='g_theta')(attention_inputs,is_training=is_training)
        else:
            main_theta = main_inputs
            attention_theta = attention_inputs
            g_theta = attention_inputs
        
        height = main_theta.shape.as_list()[1]
        width = main_theta.shape.as_list()[2]
        dim_inner = main_theta.shape.as_list()[-1]

        if self._batch_size is None:
            self._batch_size = -1

        main_theta = tf.reshape(main_theta,[self._batch_size,height * width,dim_inner],name='main_theta_reshape')
        attention_theta = tf.reshape(attention_theta,[self._batch_size,attention_theta.shape.as_list()[1] * attention_theta.shape.as_list()[2],dim_inner],name='attention_theta_reshape')
        g_theta = tf.reshape(attention_theta,[self._batch_size,g_theta.shape.as_list()[1] * g_theta.shape.as_list()[2],dim_inner],name='attention_theta_reshape')

        main_attention_theta = tf.matmul(main_theta,attention_theta,transpose_a=False,transpose_b=True,name='main_attention_theta_matmul')

        tf.Summary.Image('attention map',main_attention_theta)
        if self._use_softmax:
            p = tf.nn.softmax(main_attention_theta,name='main_attention_theta_softmax')
        else:
            ones = tf.constant(1,shape=main_attention_theta.shape)
            ones = tf.reduce_sum(ones)

            zeros = tf.constant(0,shape=main_attention_theta.shape)
            denom = tf.add(zeros,ones)

            tf.stop_gradient(denom)

            p = tf.div(main_attention_theta,denom,name='main_attention_theta_dot_product')
        
        main_attention = tf.matmul(p,g_theta)
        main_attention = tf.reshape(main_attention,[self._batch_size,height,width,dim_inner])

        if self._use_conv:
            main_attention = Unit2D(dim_inner,activation_fn=None,use_batch_norm=False)(main_attention,is_training=is_training)

        main_attention = snt.BatchNormV2(scale=True)(main_attention,is_training=is_training,test_local_stats=False)

        return main_attention

class channel_nonlocal_fusion(snt.AbstractModule):

    def __init__(self,batch_size = None,
                 use_softmax = True,
                 name = 'channel_nonlocal_fusion'):
        super(channel_nonlocal_fusion, self).__init__(name=name)
        self._batch_size = batch_size
        self._use_softmax = use_softmax

    def _build(self, main_inputs , attention_inputs , is_training):
        
       
        main_theta = main_inputs
        attention_theta = attention_inputs
        
        height = main_theta.shape.as_list()[1]
        width = main_theta.shape.as_list()[2]
        dim_inner = main_theta.shape.as_list()[-1]

        if self._batch_size is None:
            self._batch_size = -1

        main_theta = tf.reshape(main_theta,[self._batch_size,dim_inner,height * width],name='main_theta_reshape')
        
        attention_theta = tf.reshape(attention_theta,
            [self._batch_size,dim_inner,attention_theta.shape.as_list()[1] * attention_theta.shape.as_list()[2],],name='attention_theta_reshape')

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
        main_attention = tf.reshape(main_attention,[self._batch_size,height,width,dim_inner])

        main_attention = snt.BatchNormV2(scale=True)(main_attention,is_training=is_training,test_local_stats=False)

        return main_attention

            
if __name__ == "__main__":
    s1 = tf.placeholder(tf.float32,[None,224,224,3])
    s2 = tf.placeholder(tf.float32,[None,224,224,3])
    a = space_nonlocal_fusion(256)(s1,s2,True)
    for i in tf.global_variables():
        print(i)
    
