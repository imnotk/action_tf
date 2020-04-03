from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from model import TS_resnet , ts_inception_v1
from model import ts_fusion_net
from model import ts_non_local_fusion_net
import os
import time
import numpy as np
import ucf_ts,hmdb_ts
import pickle
from sklearn.manifold import TSNE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gpu_options = tf.GPUOptions (per_process_gpu_memory_fraction=0.98, allow_growth=True)
file_save_path = '/mnt/zhujian/action_recognition'
log_dir = 'log_dir_with_two_stream_m2d/'
tf.app.flags.DEFINE_string ('video_dir', './UCF-101/', 'the orginal and optical video directory')
tf.app.flags.DEFINE_integer ('image_size', 224, 'the uniform input video size')
tf.app.flags.DEFINE_integer ('num_classes', 101, 'the classes number of the video dataset')
tf.app.flags.DEFINE_integer ('frame_counts', 10, 'the input video frame counts')
tf.app.flags.DEFINE_integer ('batch_size', 1, 'the inputed batch size ')
tf.app.flags.DEFINE_integer ('gpu_nums', 2, 'the inputed batch size ')
tf.app.flags.DEFINE_string ('dataset', 'UCF101', 'the training dataset')
tf.app.flags.DEFINE_string ('eval_type', 'fusion', 'fusion or joint')
tf.app.flags.DEFINE_string ('fusion_type','non_local','fhn or fpn or non local')
tf.app.flags.DEFINE_integer ('split', 0, 'split number ')
tf.app.flags.DEFINE_string ('model_type', 'resnet50', 'use which model vgg,inception,resnet and so on')
tf.app.flags.DEFINE_string ('fusion_mode', 'add', 'use add or concat to fusion rgb and flow feature')
tf.app.flags.DEFINE_string ('test_crop', 'center', 'use center or multiscale crop')
tf.app.flags.DEFINE_integer('video_split',25,'video split')
tf.app.flags.DEFINE_boolean ('use_space', True, 'use space time fusion or not')


Flags = tf.app.flags.FLAGS

split = Flags.split
eval_type = Flags.eval_type
gpu_nums = Flags.gpu_nums
model_type = Flags.model_type
fusion_type = Flags.fusion_type
test_crop = Flags.test_crop
fusion_mode = Flags.fusion_mode
video_split = Flags.video_split
test_crop = Flags.test_crop
space = Flags.use_space
batch_size = Flags.batch_size
_FRAME_COUNTS = Flags.frame_counts
_IMAGE_SIZE = Flags.image_size
_NUM_CLASSES = Flags.num_classes
dataset_name = Flags.dataset

if test_crop == 'multi':
    factor = 10
else:
    factor = 2



rgb_train = tf.placeholder (tf.float32,
                                    [None, _FRAME_COUNTS * factor, _IMAGE_SIZE, _IMAGE_SIZE, 3])

flow_train = tf.placeholder (tf.float32,
                                    [None, _FRAME_COUNTS * factor, _IMAGE_SIZE, _IMAGE_SIZE, 20])

y_ = tf.placeholder (tf.float32, [None, _NUM_CLASSES])

dataset = Flags.dataset
if dataset == 'UCF101':
    dataset = ucf_ts.ucf_dataset (split_number=split, is_training_split=False,
                                                batch_size=batch_size, epoch=1,test_crop = test_crop,
                                                eval_type=eval_type, frame_counts=_FRAME_COUNTS,
                                                image_size=_IMAGE_SIZE,
                                                prefetch_buffer_size=batch_size).test_dataset ()
elif dataset == 'hmdb51':
    dataset = hmdb_ts.hmdb_dataset (split_number=split, is_training_split=False,
                                                batch_size=batch_size, epoch=1,test_crop = test_crop,
                                                eval_type=eval_type, frame_counts=_FRAME_COUNTS,
                                                image_size=_IMAGE_SIZE,
                                                prefetch_buffer_size=batch_size).test_dataset ()
    
iter = dataset.make_initializable_iterator ()
next_element = iter.get_next ()

if model_type == 'resnet50':
    base_net = 'TS_resnet50'
elif model_type == 'resnet101':
    base_net = 'ts_resnet101'
elif model_type == 'inception_v1':
    base_net = 'ts_inception_v1'


with tf.variable_scope ('RGB', reuse=tf.AUTO_REUSE):
    if model_type == 'resnet50':
        rgb_model = TS_resnet.Resnet(num_classes=_NUM_CLASSES,final_endpoint='logits',name='resnet_v1_50')
        print ('resnet50 process successfully')
    elif model_type == 'resnet101':
        rgb_model = TS_resnet.Resnet(num_classes=_NUM_CLASSES,final_endpoint='logits',name='resnet_v1_101',unit_num=[3, 4, 23, 3])
    elif model_type == 'inception_v1':
        rgb_model = ts_inception_v1.InceptionV1(num_classes=_NUM_CLASSES)

with tf.variable_scope ('Flow', reuse=tf.AUTO_REUSE):
    if model_type == 'resnet50':
        flow_model = TS_resnet.Resnet(num_classes=_NUM_CLASSES,final_endpoint='logits',name='resnet_v1_50')
        print ('resnet50 process successfully')
    elif model_type == 'resnet101':
        flow_model = TS_resnet.Resnet(num_classes=_NUM_CLASSES,final_endpoint='logits',name='resnet_v1_101',unit_num=[3, 4, 23, 3])
    elif model_type == 'inception_v1':
        flow_model = ts_inception_v1.InceptionV1(num_classes=_NUM_CLASSES)

with tf.variable_scope('Fusion',reuse=tf.AUTO_REUSE):
    if fusion_type == 'fhn':
        fusion_model = ts_fusion_net.FeatureHierachyNetwork (num_classes=_NUM_CLASSES,fusion_mode = fusion_mode)
    elif fusion_type == 'non_local':
        fusion_model = ts_non_local_fusion_net.space_cross_correlation_Network(num_classes=_NUM_CLASSES,use_space=space,fusion_mode = fusion_mode)
    elif fusion_type == 'channel':
        fusion_model = ts_non_local_fusion_net.channel_cross_correlation_Network(num_classes=_NUM_CLASSES,fusion_mode = fusion_mode)
    elif fusion_type == 'space_channel':
        fusion_model = ts_non_local_fusion_net.cross_correlation_Network(num_classes=_NUM_CLASSES,fusion_mode = fusion_mode)


rgb_logits, rgb_endpoints = rgb_model (rgb_train, is_training=False,
                                            dropout_keep_prob=1.0)

flow_logits, flow_endpoints = flow_model (flow_train, is_training=False,
                                            dropout_keep_prob=1.0)
print(rgb_endpoints)
if model_type in ['resnet50','resnet101','resnet152']:
    no_relu_rgb_endpoints = {}
    no_relu_flow_endpoints = {}
    for k,v in rgb_endpoints.items():
        if 'no_relu' in k:
            no_relu_rgb_endpoints[k] = v
    for k,v in flow_endpoints.items():
        if 'no_relu' in k:
            no_relu_flow_endpoints[k] = v
    feature_list = [no_relu_rgb_endpoints,no_relu_flow_endpoints]
else:
    feature_list = [rgb_endpoints,flow_endpoints]
fusion_logits, endpoints = fusion_model (feature_list, is_training=False, 
                                            dropout_keep_prob=1.0)

if eval_type == 'joint':
    model_logits = rgb_logits + 1.5 * flow_logits + 2 * fusion_logits
elif eval_type == 'fusion':
    model_logits = fusion_logits

print('model init successfully')

rgb_var_map = {}
for var in tf.global_variables ():
    if var.name.split ('/')[0] == 'RGB':
            rgb_var_map[var.name.replace (':0', '')] = var
rgb_saver = tf.train.Saver (var_list=rgb_var_map, reshape=True)

flow_var_map = {}
for var in tf.global_variables ():
    if var.name.split ('/')[0] == 'Flow':
            flow_var_map[var.name.replace (':0', '')] = var
flow_saver = tf.train.Saver (var_list=flow_var_map, reshape=True)

fusion_var_map = {}
for var in tf.global_variables ():
    if var.name.split ('/')[0] == 'Fusion':
        fusion_var_map[var.name.replace (':0', '')] = var
fusion_saver = tf.train.Saver (var_list=fusion_var_map, reshape=True)

saver = tf.train.Saver (max_to_keep=15, reshape=True)

train_log_path = os.path.join (log_dir, dataset_name, str (split), base_net, eval_type,fusion_type,fusion_mode)

        
model_predictions = tf.nn.softmax (model_logits)
correct_prediction = tf.equal (tf.argmax (model_predictions, 1), tf.argmax (y_, 1))
train_accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))

init_op = tf.global_variables_initializer ()
merge_summary = tf.summary.merge_all ()
sess = tf.Session (config=tf.ConfigProto (gpu_options=gpu_options,allow_soft_placement=True))
sess.run (init_op)

path = os.path.join ('/mnt/zhujian/action_recognition/ts_m2d/', dataset_name, str (split),
                                        base_net,eval_type,fusion_type,fusion_mode,'final')
ckpt = tf.train.get_checkpoint_state (path)
if ckpt is not None:
    saver.restore (sess, ckpt.model_checkpoint_path)
    print('checkpoint restore successfully')

sess.run (iter.initializer)
feed_dict = {}
test_batch_accuracy = []
valid_accuracy = []
fusion_video_predict = []
rgb_video_predict = []
flow_video_predict = []
k = 1

while True:
    try:
        t = time.time()
        test_rgb, test_flow, test_label = sess.run (next_element)
        print(test_rgb.shape)
        test_rgb = np.reshape(test_rgb,[batch_size,1,-1,224,224,3])
        test_flow = np.reshape(test_flow,[batch_size,1,-1,224,224,20])
        l1 = []
        l2 = []
        l3 = []

        feed_dict[y_] = test_label
        duratio = time.time()
        for i in range(batch_size):
            feed_dict[rgb_train] = test_rgb[i,...]
            feed_dict[flow_train] = test_flow[i,...]
            fusion_logits, rgb_logits, flow_logits = sess.run([
                endpoints['last'],rgb_endpoints['last'],flow_endpoints['last'] ],feed_dict=feed_dict)
            # fusion_logit = sess.run(endpoints['last'],feed_dict=feed_dict)
            fusion_logits = np.sum(fusion_logits,axis=1)
            rgb_logits = np.sum(rgb_logits,axis=1)
            flow_logits = np.sum(flow_logits,axis=1)
            l1.append(fusion_logits)
            l2.append(rgb_logits)
            l3.append(flow_logits)
        end_t = time.time() 

        l1 = np.array(l1)
        l2 = np.array(l2)
        l3 = np.array(l3)
        
        print(rgb_logits.shape)
        for i in range(fusion_logits.shape[0]):
            fusion_video_predict.append((test_label[i],l1[i]))
            rgb_video_predict.append((test_label[i],l2[i]))
            flow_video_predict.append((test_label[i],l3[i]))

        print(len(fusion_video_predict))
        k +=1

        del test_rgb , test_flow , test_label          
    except tf.errors.OutOfRangeError:
        
        with open(os.path.join(train_log_path,'fusion_video_tsne_%s.pickle' % test_crop),'wb') as f:
                        pickle.dump(fusion_video_predict,f)
                        print('video predict result save successfully')
        
        with open(os.path.join(train_log_path,'rgb_video_tsne_%s.pickle' % test_crop),'wb') as f:
                        pickle.dump(rgb_video_predict,f)
                        print('video predict result save successfully')
        
        with open(os.path.join(train_log_path,'flow_video_tsne_%s.pickle' % test_crop),'wb') as f:
                        pickle.dump(flow_video_predict,f)
                        print('video predict result save successfully')

        break

