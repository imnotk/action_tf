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
    'w':tf.contrib.layers.l2_regularizer(scale=5e-4)
}


def _build_feature(total_endpoints):

        # we don't need the first two conv feature map due to low semantic information
        rgb_endpoints = total_endpoints
        feature_list = []

        rgb_56 = [];flow_56 = []
        rgb_28 = [];flow_28 = []
        rgb_14 = [];flow_14 = []
        rgb_7 = [];flow_7 = []

        for k , v in rgb_endpoints.items():
            
            if 'pool' in k or 'logits' in k:
                continue

            if v.shape.as_list()[2] == 56:
                rgb_56.append(v)
            if v.shape.as_list()[2] == 28:
                rgb_28.append(v)
            if v.shape.as_list()[2] == 14:
                rgb_14.append(v)
            if v.shape.as_list()[2] == 7:
                rgb_7.append(v)

        rgb_56_feature = tf.concat(rgb_56,axis=-1)
        rgb_28_feature = tf.concat(rgb_28,axis=-1)
        rgb_14_feature = tf.concat(rgb_14,axis=-1)
        rgb_7_feature = tf.concat(rgb_7,axis=-1)

        

        feature_list.append(rgb_56_feature)
        feature_list.append(rgb_28_feature)
        feature_list.append(rgb_14_feature)
        feature_list.append(rgb_7_feature)
        return feature_list


class spacetime_nonlocal_fusion(snt.AbstractModule):

    def __init__(self,output_channels = None,
                 batch_size = None,
                 use_max_pooling = True,
                 use_conv = True,
                 use_batch_norm = False,
                 use_softmax = True,
                 activation_fn = None,
                 fusion_mode = 'concat',
                 name = 'space_nonlocal_fusion'):
        super(spacetime_nonlocal_fusion, self).__init__(name=name)
        self._output_channels = output_channels
        self._batch_size = batch_size
        self._use_max_pooling = use_max_pooling
        self._use_conv = use_conv
        self._use_softmax = use_softmax
        self._fusion_model = fusion_mode
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn


    def _build(self, inputs , is_training):
        
        # inputs already has same shape as (None,frame_counts,56,56,256)

        feature_dimension = inputs.shape.as_list()[-1]

        theta = Unit3d(output_channels=feature_dimension // 2,use_batch_norm=self._use_batch_norm,activation_fn=self._activation_fn,name='theta')(inputs,is_training=is_training)
        
        
        if self._use_max_pooling:
            inputs_pooling = tf.nn.max_pool3d(inputs,ksize=(1,1,2,2,1),strides=(1,1,2,2,1),padding='SAME')
            beta = Unit3d(output_channels=feature_dimension // 2,use_batch_norm=self._use_batch_norm,activation_fn=self._activation_fn,name='beta')(inputs_pooling,is_training=is_training)
            g =  Unit3d(output_channels=feature_dimension // 2,use_batch_norm=self._use_batch_norm,activation_fn=self._activation_fn,name='g')(inputs_pooling,is_training=is_training)
        else:
            beta = Unit3d(output_channels=feature_dimension // 2,use_batch_norm=self._use_batch_norm,activation_fn=self._activation_fn,name='beta')(inputs,is_training=is_training)
            g =  Unit3d(output_channels=feature_dimension // 2,use_batch_norm=self._use_batch_norm,activation_fn=self._activation_fn,name='g')(inputs,is_training=is_training)

        if self._batch_size is None:
            self._batch_size = -1

        theta_re = tf.reshape(theta,[self._batch_size, 
                    theta.shape.as_list()[1] *  theta.shape.as_list()[2] * theta.shape.as_list()[3],
                    theta.shape.as_list()[-1]],name='theta_reshape')

        beta_re = tf.reshape(beta,[self._batch_size, 
                    beta.shape.as_list()[1] * beta.shape.as_list()[2] * beta.shape.as_list()[3],
                    beta.shape.as_list()[-1]],name='beta_reshape')

        g_re = tf.reshape(g,[self._batch_size, 
                    g.shape.as_list()[1] *  g.shape.as_list()[2] * g.shape.as_list()[3],
                    g.shape.as_list()[-1]],name='g_reshape')

        theta_beta = tf.matmul(theta_re,beta_re,transpose_a=False,transpose_b=True)
        
        if self._use_softmax:
            p_1 = tf.nn.softmax(theta_beta,name='theta_beta_softmax')
        else:
            ones = tf.constant(1,shape=theta_beta.shape)
            ones = tf.reduce_sum(ones)

            zeros = tf.constant(0,shape=theta_beta.shape)
            denom = tf.add(zeros,ones)

            tf.stop_gradient(denom)

            p_1 = tf.div(theta_beta,denom,name='theta_beta_dot_product')

        t = tf.matmul(p_1,g_re,name='non_local_matmul')

        t_re = tf.reshape(t,[self._batch_size, 
                    theta.shape.as_list()[1], theta.shape.as_list()[2], theta.shape.as_list()[3],
                    g.shape.as_list()[-1]],name='t_reshape')

        blob_out = Unit3d(output_channels=feature_dimension,use_batch_norm=False,activation_fn=None,name='blob_out')(t_re,is_training=is_training)

        blob_out = blob_out + inputs

        return blob_out

            
if __name__ == "__main__":
    s1 = tf.placeholder(tf.float32,[None,25,56,56,256])
    a = spacetime_nonlocal_fusion(256)(s1,True)
    print(a)
