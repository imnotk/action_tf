from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from model import i3d,Res3D,TS_resnet
from model import ts_non_local_net
import os
import time
import numpy as np
import ucf_ts,ucf_ts_val
import pickle

slim = tf.contrib.slim
os.environ[' CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gpu_options = tf.GPUOptions (per_process_gpu_memory_fraction=0.98, allow_growth=True)
file_save_path = '/mnt/zhujian/action_recognition'
log_dir = 'log_dir_with_two_stream_3d_intra_fusion/'
# base_net = 'resnet_v1_50'
tf.app.flags.DEFINE_string ('video_dir', './UCF-101/', 'the orginal and optical video directory')
tf.app.flags.DEFINE_integer ('image_size', 224, 'the uniform input video size')
tf.app.flags.DEFINE_integer ('num_classes', 101, 'the classes number of the video dataset')
tf.app.flags.DEFINE_integer ('frame_counts', 1, 'the input video frame counts')
tf.app.flags.DEFINE_integer ('batch_size', 1, 'the inputed batch size ')
tf.app.flags.DEFINE_integer ('gpu_nums', 2, 'the inputed batch size ')
tf.app.flags.DEFINE_string ('dataset', 'UCF101', 'the training dataset')
tf.app.flags.DEFINE_string ('eval_type', 'fusion', 'fusion or joint')
tf.app.flags.DEFINE_string ('fusion_type','non_local','fhn or fpn or non local')
tf.app.flags.DEFINE_integer ('split', 0, 'split number ')
tf.app.flags.DEFINE_boolean ('use_pbn', False, 'use partial batch norm')
tf.app.flags.DEFINE_string ('model_type', 'resnet50', 'use which model vgg,inception,resnet and so on')
tf.app.flags.DEFINE_string ('fusion_mode', 'add', 'use add or concat to fusion rgb and flow feature')
tf.app.flags.DEFINE_string ('test_crop', 'multi', 'use center or multiscale crop')
tf.app.flags.DEFINE_string ('phase', 'test', 'use train or test phase ')
tf.app.flags.DEFINE_integer('video_split',25,'video split')
tf.app.flags.DEFINE_boolean ('use_batch_norm', False, 'use fusion batch norm')

Flags = tf.app.flags.FLAGS

split = Flags.split
use_pbn = Flags.use_pbn
eval_type = Flags.eval_type
gpu_nums = Flags.gpu_nums
model_type = Flags.model_type
fusion_type = Flags.fusion_type
test_crop = Flags.test_crop
fusion_mode = Flags.fusion_mode
use_batch_norm = Flags.use_batch_norm
video_split = Flags.video_split
test_crop = Flags.test_crop
phase = Flags.phase

if use_batch_norm:
    print('use batch norm')
else:
    print('use batch norm is false')

class model ():
    def __init__(self,
                 video_dir='./UCF-101/',
                 image_size=224,
                 num_classes=101,
                 frame_counts=64,
                 batch_size=3,
                 learning_rate=1e-4,
                 num_segments=3,
                 TRAINING_STEP=10000,
                 epoch=10,
                 dataset='UCF101',
                 reboot=False):
        self._video_dir = video_dir
        self._IMAGE_SIZE = image_size
        self._NUM_CLASSES = num_classes
        self._FRAME_COUNTS = frame_counts
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.TRAINING_STEP = TRAINING_STEP
        self.num_segments = num_segments
        self.epoch = epoch
        self.dataset_name = dataset
        self.reboot = reboot

        self.init_config ()
        self.init_type ()
        self.init_dataset ()
        self.init_model ()
        self.train ()
        self._init_sess ()
        self._restore_model ()

    def init_config(self):

        if eval_type in ['rgb', 'joint']:
            self.rgb_train = tf.placeholder (tf.float32,
                                             [None, self._FRAME_COUNTS, self._IMAGE_SIZE, self._IMAGE_SIZE, 3])
        else:
            self.rgb_train = None

        if eval_type in ['flow', 'joint']:
            self.flow_train = tf.placeholder (tf.float32,
                                              [None, self._FRAME_COUNTS, self._IMAGE_SIZE, self._IMAGE_SIZE, 20])
        else:
            self.flow_train = None
        

        # init label
        self.y_ = tf.placeholder (tf.float32, [None, self._NUM_CLASSES])
        # init optimizer
        
    def init_dataset(self):
        dataset = Flags.dataset
        if dataset == 'UCF101':
            self.dataset = ucf_ts.ucf_dataset (split_number=split, is_training_split=False,
                                                      batch_size=self.batch_size, epoch=1,test_crop=test_crop,
                                                      eval_type=eval_type,
                                                      image_size=self._IMAGE_SIZE,frame_counts=self._FRAME_COUNTS,
                                                      prefetch_buffer_size=self.batch_size).test_dataset ()
            
        elif dataset == 'hmdb51':
            self.dataset = hmdb_video_2d.hmdb_dataset (split_number=split, is_training_split=False,
                                                       batch_size=self.batch_size, epoch=1,
                                                       eval_type=eval_type,
                                                       image_size=self._IMAGE_SIZE,
                                                       prefetch_buffer_size=self.batch_size).test_dataset ()
           

        iter = tf.data.Iterator.from_structure (self.dataset.output_types, self.dataset.output_shapes)
        self.next_element = iter.get_next ()
        self.training_init_op = iter.make_initializer (self.dataset)


    def init_type(self):
        if model_type == 'resnet50':
            self.base_net = 'TS_resnet50'

    def init_model(self):
        
        if eval_type in ['rgb', 'joint']:

            with tf.variable_scope ('RGB', reuse=tf.AUTO_REUSE):
                if model_type == 'resnet50':
                    self.rgb_model = TS_resnet.Resnet(num_classes=self._NUM_CLASSES,final_endpoint='logits',name='resnet_v1_50')
                    print ('resnet50 process successfully')

                with tf.variable_scope('Fusion',reuse=tf.AUTO_REUSE):
                    if fusion_type == 'non_local':
                        self.rgb_fusion_model = ts_non_local_net.FeatureHierachyNetwork(num_classes=self._NUM_CLASSES,fusion_mode = fusion_mode,use_batch_norm = use_batch_norm)

        if eval_type in ['flow', 'joint']:

            with tf.variable_scope ('Flow', reuse=tf.AUTO_REUSE):
                if model_type == 'resnet50':
                    self.flow_model = TS_resnet.Resnet(num_classes=self._NUM_CLASSES,final_endpoint='logits',name='resnet_v1_50')
                    print ('resnet50 process successfully')
                with tf.variable_scope('Fusion',reuse=tf.AUTO_REUSE):
                    if fusion_type == 'non_local':
                        self.flow_fusion_model = ts_non_local_net.FeatureHierachyNetwork(num_classes=self._NUM_CLASSES,fusion_mode = fusion_mode,use_batch_norm = use_batch_norm,phase=phase)
        
        

    @staticmethod
    def Multigpu_train(model_fn,num_gpus,rgb_input,flow_input):

        in_splits = {}
        in_splits['rgb'] = tf.split(rgb_input,num_gpus) if rgb_input is not None else None
        in_splits['flow'] = tf.split(flow_input,num_gpus) if flow_input is not None else None

        out_split = []
        for i in range(num_gpus):
            if tf.test.is_built_with_cuda():
                device_type = 'GPU'
            else:
                device_type = 'CPU'
            with tf.device(tf.DeviceSpec(device_type=device_type,device_index=i)):
                with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
                    if in_splits['flow'] is None:
                        out_split.append(model_fn(in_splits['rgb'][i],None))
                    elif in_splits['rgb'] is None:
                        out_split.append(model_fn(None,in_splits['flow'][i]))
                    else:
                        out_split.append (model_fn (in_splits['rgb'][i], in_splits['flow'][i]))
                    tf.get_variable_scope().reuse_variables()
        out = tf.concat(out_split,axis=0)
        return out

    def train_model(self,rgb_input,flow_input):

        
        if eval_type in ['rgb', 'joint'] and rgb_input is not None:
            rgb_logits, rgb_endpoints = self.rgb_model (rgb_input, is_training=False,
                                                      dropout_keep_prob=1.0)
            rgb_fusion_logits, endpoints = self.rgb_fusion_model (rgb_endpoints,is_training=False,
                                                            dropout_keep_prob=1.0)

        if eval_type in ['flow', 'joint'] and flow_input is not None:
            flow_logits, flow_endpoints = self.flow_model (flow_input, is_training=False,
                                                        dropout_keep_prob=1.0)
            flow_fusion_logits, endpoints = self.flow_fusion_model (flow_endpoints,is_training=False,
                                                            dropout_keep_prob=1.0)

        if eval_type == 'rgb':
            model_logits = rgb_fusion_logits
        elif eval_type == 'flow':
            model_logits = flow_fusion_logits
        elif eval_type == 'joint':
            model_logits = rgb_fusion_logits + flow_fusion_logits
        return model_logits

    def get_saver(self):

        if eval_type in ['rgb', 'joint'] :
            rgb_var_map = {}
            for var in tf.global_variables ():
                if var.name.split ('/')[0] == 'RGB' and var.name.split ('/')[1] != 'Fusion':
                        rgb_var_map[var.name.replace (':0', '')] = var
            self.rgb_saver = tf.train.Saver (var_list=rgb_var_map, reshape=True)

        if eval_type in ['flow', 'joint'] :
            flow_var_map = {}
            for var in tf.global_variables ():
                if var.name.split ('/')[0] == 'Flow' and var.name.split ('/')[1] != 'Fusion':
                        flow_var_map[var.name.replace (':0', '')] = var
            self.flow_saver = tf.train.Saver (var_list=flow_var_map, reshape=True)

        self.fusion_var_map = {}
        for var in tf.global_variables ():
            if var.name.split ('/')[1] == 'Fusion':
                self.fusion_var_map[var.name.replace (':0', '')] = var
        self.fusion_saver = tf.train.Saver (var_list=self.fusion_var_map, reshape=True)

        self.saver = tf.train.Saver (max_to_keep=15, reshape=True)
        self.best_saver = tf.train.Saver (reshape=True)

        self.rgb_regular_loss = tf.losses.get_regularization_loss (scope='RGB')
        self.flow_regular_loss = tf.losses.get_regularization_loss (scope='Flow')
        self.fusion_regular_loss = tf.losses.get_regularization_loss (scope='Fusion')
        if eval_type == 'fusion':
            self.regular_loss = self.fusion_regular_loss
        else:
            self.regular_loss = self.rgb_regular_loss + self.flow_regular_loss + self.fusion_regular_loss

    def train(self):
        if gpu_nums == 1:
            self.model_logits = self.train_model (self.rgb_train,self.flow_train)
        else:
            self.model_logits = self.Multigpu_train(self.train_model,gpu_nums,self.rgb_train,self.flow_train)

        self.get_saver()

        self.model_predictions = tf.nn.softmax (self.model_logits)
        correct_prediction = tf.equal (tf.argmax (self.model_predictions, 1), tf.argmax (self.y_, 1))
        self.train_accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))

    def _init_sess(self):
        self.init_op = tf.global_variables_initializer ()
        self.merge_summary = tf.summary.merge_all ()
        self.sess = tf.Session (config=tf.ConfigProto (gpu_options=gpu_options,allow_soft_placement=True))
        self.sess.run (self.init_op)
        log_path = os.path.join (file_save_path, log_dir, self.dataset_name, str (split), self.base_net,
                                 eval_type,fusion_type,fusion_mode)
        if os.path.exists (log_path) is False:
            os.makedirs (log_path)
        self.train_writer = tf.summary.FileWriter (
            log_path, self.sess.graph)

        self.train_log_path = os.path.join (log_dir, self.dataset_name, str (split), self.base_net, eval_type,fusion_type,fusion_mode)
        if os.path.exists (self.train_log_path) is False:
            os.makedirs (self.train_log_path)


    def _restore_model(self):
        
            path = os.path.join ('/mnt/zhujian/action_recognition/3d/', self.dataset_name, str (split),
                                        self.base_net,eval_type,fusion_type,fusion_mode)
            ckpt = tf.train.get_checkpoint_state (path)
            if ckpt is not None:
                self.saver.restore (self.sess, ckpt.model_checkpoint_path)

    def one_epoch_test(self, epoch):
        self.sess.run (self.training_init_op)
        feed_dict = {}
        test_batch_accuracy = []
        valid_accuracy = []
        video_predict = []
        k = 1
        while  True:
            try:
                t = time.time()
                if eval_type == 'rgb':
                    test_rgb,  test_label = self.sess.run (self.next_element)
                elif eval_type == 'flow':
                    test_flow, test_label = self.sess.run (self.next_element)
                else:
                    test_rgb, test_flow, test_label = self.sess.run (self.next_element)
                logits = []
                feed_dict[self.y_] = test_label
                duratio = time.time() - t
                for i in range(self.batch_size):
                    if eval_type in ['rgb','joint']:
                        feed_dict[self.rgb_train] = test_rgb[i,...]
                    if eval_type in ['flow','joint']:
                        feed_dict[self.flow_train] = test_flow[i,...]
                    out_logits = self.sess.run(self.model_logits,feed_dict=feed_dict)
                    # out_logits = sess.run([model_predictions],feed_dict=feed_dict)

                    out_logits = np.sum(out_logits,axis=0)
                    logits.append(out_logits)

                end_t = time.time() 
                logits = np.array(logits)
                for i in range(logits.shape[0]):
                    video_predict.append((test_label[i],logits[i]))

                out_pred = (np.equal (np.argmax (logits, 1), np.argmax (test_label, 1))).sum() / self.batch_size
                test_batch_accuracy.append(out_pred)
                end_t = time.time() - t
                print ('after %d step with %.2fs duratio,the test accuracy is %f,use time %.2fs' % 
                    (k, duratio,np.mean (np.array (test_batch_accuracy)),end_t))
                k +=1
                
            except tf.errors.OutOfRangeError:
                with open(os.path.join(self.train_log_path,'test_result_%s.txt' % test_crop),'a') as f:
                    f.write(str(np.mean (np.array (test_batch_accuracy)))+'\n')
                
                with open(os.path.join(self.train_log_path,'video_predict_%s.pickle' % test_crop),'wb') as f:
                        pickle.dump(video_predict,f)
                        print('video predict result save successfully')


                break

    def main(self):
            self.one_epoch_test (0)



def main(argv):
    m = model (video_dir=Flags.video_dir,
               image_size=Flags.image_size,
               num_classes=Flags.num_classes,
               frame_counts=Flags.frame_counts,
               batch_size=Flags.batch_size,
               dataset=Flags.dataset)
    m.main ()


if __name__ == '__main__':
    tf.app.run (main)
