from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf
from keras.applications.densenet import DenseNet121,DenseNet169


initializer = {
    'w':tf.contrib.layers.xavier_initializer()
}
regularizers = {
    'w':tf.contrib.layers.l2_regularizer(scale=5e-4)
}




def conv_block(inputs,growth_rate,is_training,name):
    bn1 = snt.BatchNormV2(scale=True,name=name+'_0_bn')
    net = bn1(inputs,is_training=is_training,test_local_stats=False)
    net = tf.nn.relu(net)
    net = snt.Conv2D(4*growth_rate,kernel_shape=(1,1),name=name+'_1_conv',use_bias=False,
                    regularizers=regularizers)(net)

    bn2 = snt.BatchNormV2(scale=True,name=name+'_1_bn')
    net = bn2(net,is_training=is_training,test_local_stats=False)
    net = tf.nn.relu(net)
    net = snt.Conv2D(growth_rate,kernel_shape=(3,3),name=name+'_2_conv',use_bias=False,
                    regularizers=regularizers)(net)
    output = tf.concat([inputs,net],axis=-1)
    return output

def dense_block(x,blocks,is_training,name):

    for i in range(blocks):
        x = conv_block(x,32,is_training,name=name+'_block'+str(i+1))

    return x

def transition_block(inputs,reduction,is_training,name):
    bn = snt.BatchNormV2(scale=True,name=name+'_bn')
    net = bn(inputs,is_training=is_training,test_local_stats=False)
    net = tf.nn.relu(net)
    net = snt.Conv2D(net.shape.as_list()[-1] * reduction,kernel_shape=(1,1),use_bias=False,name=name+'_conv')(net)
    net = tf.nn.avg_pool(net,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
    return net

class DenseNet(snt.AbstractModule):
    
    VALID_ENDPOINTS = (
        'conv1',
        'pool1',
        'dense1',
        'trans1',
        'dense2',
        'trans2',
        'dense3',
        'trans3',
        'dense4',
        'trans4',
        'logits',
        'Predictions'
    )

    def  __init__(self,num_classes = 1000,spatia_squeeze = True,eval_type='rgb',unit_num = [6,12,24,16],use_pbn = False,filters=32,
                 final_endpoint = 'logits',name = 'densenet121'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(DenseNet, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint
        self._eval_type = eval_type
        self._unit_num = unit_num
        self._filters = filters
        self.use_pbn = use_pbn
    

    def _build(self,inputs,is_training,dropout_keep_prob=0.5):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        
        net = inputs
        end_points = {}
        end_point = 'conv1'
        net = snt.Conv2D(64,kernel_shape=(7,7),stride=(2,2),name='conv1',use_bias=False,
                    regularizers=regularizers)(net)
        net = snt.BatchNormV2(scale=True,name='conv1/bn')(net,is_training=is_training,test_local_stats=False)
        net = tf.nn.relu(net)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        
        end_point = 'pool1'
        net = tf.nn.max_pool(net,ksize=(1,3,3,1),strides=(1,2,2,1),
                               padding=snt.SAME,name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'dense1'
        num_units = self._unit_num[0]
        net = dense_block(net,num_units,is_training=is_training,name='conv2')    
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        
        end_point = 'trans1'
        net = transition_block(net,0.5,is_training=is_training,name='pool2')
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points

        end_point = 'dense2'
        num_units = self._unit_num[1]
        net = dense_block(net,num_units,is_training=is_training,name='conv3')    
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        

        end_point = 'trans2'
        net = transition_block(net,0.5,is_training=is_training,name='pool3')
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points

        end_point = 'dense3'
        num_units = self._unit_num[2]
        net = dense_block(net,num_units,is_training=is_training,name='conv4')    
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        
        end_point = 'trans3'
        net = transition_block(net,0.5,is_training=is_training,name='pool4')
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points

        end_point = 'dense4'
        num_units = self._unit_num[3]
        net = dense_block(net,num_units,is_training=is_training,name='conv5')    
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        
        end_point = 'trans4'
        bn = snt.BatchNormV2(scale=True,name='bn')
        net = bn(net,is_training=is_training,test_local_stats=False)
        net = tf.nn.relu(net)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points


        end_point = 'logits'
        height = net.shape.as_list()[1]
        width = net.shape.as_list()[2]
        net = tf.nn.avg_pool(net,ksize=(1,height,width,1),
                               strides=(1,1,1,1),padding=snt.VALID)
        net = tf.nn.dropout(net,keep_prob=dropout_keep_prob)
        logits = snt.Conv2D(self._num_classes,(1,1),name='logits')(net)
        if self._spatia_squeeze:
            logits = tf.squeeze(logits,(1,2),name='SpatialSqueeze')
        averge_logits = logits
        end_points[end_point] = averge_logits
        if self._final_endpoint == end_point: return averge_logits, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax(averge_logits)
        end_points[end_point] = predictions
        return predictions,end_points

if __name__ == '__main__':
    # k = Unit2d(output_channels=3)
    inputs = tf.placeholder(tf.float32,[None,224,224,3])
    with tf.variable_scope('RGB'):
        Resnet_v1 = DenseNet(num_classes=101,final_endpoint='dense1',name='densenet121')
        logits , e = Resnet_v1(inputs,is_training=True)

    for i in tf.global_variables():
        print(i)

    saver = tf.train.Saver(reshape=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,r'E:\ckpt\rgb_densenet121\model.ckpt')
      