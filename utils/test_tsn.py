from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from model import net, resnet, inception_v1, inception_v2, vgg
import os
import time
import numpy as np
import ucf_video_tsn
import hmdb_video_2d
import ucf_video_2d

slim = tf.contrib.slim
os.environ[' CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gpu_options = tf.GPUOptions (per_process_gpu_memory_fraction=0.98, allow_growth=True)
file_save_path = '/mnt/zhujian/action_recognition'
log_dir = 'log_dir_with_two_stream_2d/'
# base_net = 'resnet_v1_50'
tf.app.flags.DEFINE_string ('video_dir', './UCF-101/', 'the orginal and optical video directory')
tf.app.flags.DEFINE_integer ('image_size', 224, 'the uniform input video size')
tf.app.flags.DEFINE_integer ('num_classes', 101, 'the classes number of the video dataset')
tf.app.flags.DEFINE_integer ('frame_counts', 25, 'the input video frame counts')
tf.app.flags.DEFINE_integer ('batch_size', 8, 'the inputed batch size ')
tf.app.flags.DEFINE_integer ('gpu_nums', 2, 'the inputed batch size ')
tf.app.flags.DEFINE_float ('learning_rate', 1e-3, 'the learning rate of optimizer')
tf.app.flags.DEFINE_float ('weight_decay', 5e-4, 'the learning rate of optimizer')
tf.app.flags.DEFINE_integer ('training_step', 100000, 'the training steps of networks')
tf.app.flags.DEFINE_integer ('epoch', 100, 'the training epoches')
tf.app.flags.DEFINE_string ('dataset', 'UCF101', 'the training dataset')
tf.app.flags.DEFINE_string ('eval_type', 'rgb', 'rgb flow or joint')
tf.app.flags.DEFINE_boolean ('reboot', True, 'reboot traing process if True else False')
tf.app.flags.DEFINE_string ('lr_step', '[150,200]', 'epochs to decay learning rate by 10')
tf.app.flags.DEFINE_boolean ('is_training', True, 'is training or not')
tf.app.flags.DEFINE_integer ('split', 0, 'split number ')
tf.app.flags.DEFINE_float ('rgb_dr', 1.0, 'rgb dropout ratio')
tf.app.flags.DEFINE_float ('flow_dr', 1.0, 'flow dropout ratio')
tf.app.flags.DEFINE_integer ('num_segments', 3, 'num segments for TSN')
tf.app.flags.DEFINE_boolean ('use_pbn', False, 'use partial batch norm')
tf.app.flags.DEFINE_string ('model_type', 'inception_v2', 'use which model vgg,inception,resnet and so on')
tf.app.flags.DEFINE_boolean('tta',True,'test time augmentation')

Flags = tf.app.flags.FLAGS
tf.logging.set_verbosity (tf.logging.INFO)

rgb_dr = Flags.rgb_dr
flow_dr = Flags.flow_dr
split = Flags.split
use_pbn = Flags.use_pbn
num_segments = Flags.num_segments
eval_type = Flags.eval_type
lr_step = eval (Flags.lr_step)
gpu_nums = Flags.gpu_nums
# is_training = Flags.is_training
model_type = Flags.model_type
tta = Flags.tta

_rgb_resnet_v1_50 = 'data/rgb_snt_resnetV1_50/model.ckpt'
_flow_resnet_v1_50 = 'data/flow_snt_resnetV1_50/model.ckpt'


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
                                             [None, self._IMAGE_SIZE, self._IMAGE_SIZE, 3])
        else:
            self.rgb_train = None

        if eval_type in ['flow', 'joint']:
            self.flow_train = tf.placeholder (tf.float32,
                                              [None, self._IMAGE_SIZE, self._IMAGE_SIZE, 20])
        else:
            self.flow_train = None

        # init learning rate
        # init label
        self.y_ = tf.placeholder (tf.float32, [None, self._NUM_CLASSES])
       

    def init_dataset(self):
        dataset = Flags.dataset
        if dataset == 'UCF101':
            video = ucf_video_2d
            self.dataset = video.ucf_dataset (split_number=split, is_training_split=False,
                                                      batch_size=self.batch_size, epoch=1,
                                                      eval_type=eval_type,
                                                      image_size=self._IMAGE_SIZE,
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
            self.base_net = 'resnet_v1_50'
            self._rgb_reboot_path = 'data/rgb_snt_resnetV1_50/model.ckpt'
            self._flow_reboot_path = 'data/flow_snt_resnetV1_50/model.ckpt'
        elif model_type == 'inception_v1':
            self.base_net = 'inceptionv1'
            self._rgb_reboot_path = 'data/rgb_inception_v1/rgb_inception_v1.ckpt'
            self._flow_reboot_path = 'data/flow_inception_v1/flow_inception_v1.ckpt'
        elif model_type == 'resnet152':
            self.base_net = 'resnet_v1_152'
            self._rgb_reboot_path = 'data/rgb_resnet_v1_152/model.ckpt'
            self._flow_reboot_path = 'data/flow_resnet_v1_152/model.ckpt'
        elif model_type == 'inception_v2':
            self.base_net = 'inceptionv2'
            self._rgb_reboot_path = 'data/rgb_snt_inception_v2/model.ckpt'
            self._flow_reboot_path = 'data/flow_snt_inception_v2/model.ckpt'
        elif model_type == 'resnet18':
            self.base_net = 'resnet_v1_18'

    def init_model(self):
        if eval_type in ['rgb', 'joint']:
            with tf.variable_scope ('RGB', reuse=tf.AUTO_REUSE):
                if model_type == 'resnet50':
                    self.rgb_model = resnet.Resnet (num_classes=self._NUM_CLASSES, name='resnet_v1_50',
                                                    eval_type=eval_type)
                    print ('resnet50 process successfully')
                elif model_type == 'inception_v1':
                    self.rgb_model = inception_v1.InceptionV1 (num_classes=self._NUM_CLASSES)
                elif model_type == 'vgg_16':
                    self.rgb_model = vgg.vgg (num_classes=self._NUM_CLASSES)
                elif model_type == 'resnet152':
                    self.rgb_model = resnet.Resnet (num_classes=self._NUM_CLASSES, name='resnet_v1_152',
                                                    eval_type=eval_type,
                                                    unit_num=[3, 8, 36, 3])
                elif model_type == 'inception_v2':
                    self.rgb_model = inception_v2.InceptionV2 (num_classes=self._NUM_CLASSES)
                elif model_type == 'resnet18':
                    self.rgb_model = resnet.Vanilla_Resnet(num_classes=self._NUM_CLASSES,name='resnet_v1_18')

        if eval_type in ['flow', 'joint']:
            with tf.variable_scope ('Flow', reuse=tf.AUTO_REUSE):
                if model_type == 'resnet50':
                    self.flow_model = resnet.Resnet (num_classes=self._NUM_CLASSES, name='resnet_v1_50',
                                                     eval_type=eval_type)
                    print ('resnet50 process successfully')
                elif model_type == 'inception_v1':
                    self.flow_model = inception_v1.InceptionV1 (num_classes=self._NUM_CLASSES)
                elif model_type == 'vgg_16':
                    self.flow_model = vgg.vgg (num_classes=self._NUM_CLASSES)
                elif model_type == 'resnet152':
                    self.flow_model = resnet.Resnet (num_classes=self._NUM_CLASSES, name='resnet_v1_152',
                                                     eval_type=eval_type,
                                                     unit_num=[3, 8, 36, 3])
                elif model_type == 'inception_v2':
                    self.flow_model = inception_v2.InceptionV2 (num_classes=self._NUM_CLASSES)
                elif model_type == 'resnet18':
                    self.rgb_model = resnet.Vanilla_Resnet(num_classes=self._NUM_CLASSES,name='resnet_v1_18')

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

        if eval_type in ['flow', 'joint'] and flow_input is not None:
            flow_logits, flow_endpoints = self.flow_model (flow_input, is_training=False,
                                                        dropout_keep_prob=1.0)

        if eval_type == 'rgb':
            model_logits = rgb_logits
        elif eval_type == 'flow':
            model_logits = flow_logits
        elif eval_type == 'joint':
            model_logits = rgb_logits + 1.5 * flow_logits

        return model_logits

    def get_saver(self):
        self.saver = tf.train.Saver (max_to_keep=15, reshape=True)

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
        self.sess = tf.Session (config=tf.ConfigProto (gpu_options=gpu_options))
        self.sess.run (self.init_op)
        log_path = os.path.join (file_save_path, log_dir, self.dataset_name, str (split), 'TSN',self.base_net,
                                 eval_type)
        if os.path.exists (log_path) is False:
            os.makedirs (log_path)
        self.train_writer = tf.summary.FileWriter (
            log_path, self.sess.graph)

        self.train_log_path = os.path.join (log_dir, self.dataset_name, str (split), 'TSN',self.base_net, eval_type)
        if os.path.exists (self.train_log_path) is False:
            os.makedirs (self.train_log_path)
       
    def _restore_model(self):
            path = os.path.join ('/mnt/zhujian/action_recognition/two_stream/', self.dataset_name, str (split),'TSN',
                                        self.base_net,eval_type,'final')
            ckpt = tf.train.get_checkpoint_state (path)
            print(ckpt)
            if ckpt is not None:
                self.saver.restore (self.sess, ckpt.model_checkpoint_path)

           
    def one_epoch_test(self, epoch):
        self.sess.run (self.training_init_op)
        feed_dict = {}
        test_batch_accuracy = []
        valid_accuracy = []
        k = 1
        while  True:
            try:
                t = time.time()
                if eval_type == 'rgb':
                    test_rgb,  test_label = self.sess.run (self.next_element)
                    test_len = test_rgb.shape[1]
                elif eval_type == 'flow':
                    test_flow, test_label = self.sess.run (self.next_element)
                    test_len = test_flow.shape[1]
                else:
                    test_rgb, test_flow, test_label = self.sess.run (self.next_element)
                    test_len = test_rgb.shape[1]
                logits = []
                feed_dict[self.y_] = test_label
                duratio = time.time() - t
                for i in range(test_len):
                    try:
                        if eval_type in ['rgb','joint']:
                            feed_dict[self.rgb_train] = test_rgb[:,i,...]
                        if eval_type in ['flow','joint']:
                            feed_dict[self.flow_train] = test_flow[:,i,...]
                        out_logits = self.sess.run([self.model_logits],feed_dict=feed_dict)
                        # out_predictions = sess.run([model_predictions],feed_dict=feed_dict)
                    except:
                        raise ValueError('frame counts is not %d' % self._FRAME_COUNTS)
                    logits.append(out_logits)
                    # logits.append(out_predictions)
                try:
                    logits = np.array(logits)
                    logits = np.sum(logits,axis=0)[0]
                    out_pred = (np.equal (np.argmax (logits, 1), np.argmax (test_label, 1))).sum() / self.batch_size
                    test_batch_accuracy.append(out_pred)
                    end_t = time.time() - t
                    print ('after %d step with %.2fs duratio,the test accuracy is %f,use time %.2fs' % 
                        (k, duratio,np.mean (np.array (test_batch_accuracy)),end_t))
                    k +=1
                except:
                    break
            except tf.errors.OutOfRangeError:
                with open(os.path.join(self.train_log_path,'test_result.txt'),'w') as f:
                    f.write(str(np.mean (np.array (test_batch_accuracy))))
                break

    

    def main(self):
            self.one_epoch_test (0)


def main(argv):
    m = model (video_dir=Flags.video_dir,
               image_size=Flags.image_size,
               num_classes=Flags.num_classes,
               frame_counts=Flags.frame_counts,
               batch_size=Flags.batch_size,
               learning_rate=Flags.learning_rate,
               TRAINING_STEP=Flags.training_step,
               epoch=Flags.epoch,
               dataset=Flags.dataset,
               reboot=Flags.reboot)
    m.main ()


if __name__ == '__main__':
    tf.app.run (main)
