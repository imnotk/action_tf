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
tf.app.flags.DEFINE_integer ('frame_counts', 3, 'the input video frame counts')
tf.app.flags.DEFINE_integer ('batch_size', 8, 'the inputed batch size ')
tf.app.flags.DEFINE_integer ('gpu_nums', 2, 'the inputed batch size ')
tf.app.flags.DEFINE_float ('learning_rate', 1e-3, 'the learning rate of optimizer')
tf.app.flags.DEFINE_float ('weight_decay', 5e-4, 'the learning rate of optimizer')
tf.app.flags.DEFINE_integer ('epoch', 100, 'the training epoches')
tf.app.flags.DEFINE_string ('dataset', 'UCF101', 'the training dataset')
tf.app.flags.DEFINE_string ('eval_type', 'joint', 'fusion or joint')
tf.app.flags.DEFINE_string ('fusion_type','non_local','fhn or fpn')
tf.app.flags.DEFINE_boolean ('reboot', True, 'reboot traing process if True else False')
tf.app.flags.DEFINE_string ('lr_step', '[150,200]', 'epochs to decay learning rate by 10')
tf.app.flags.DEFINE_boolean ('is_training', True, 'is training or not')
tf.app.flags.DEFINE_integer ('split', 0, 'split number ')
tf.app.flags.DEFINE_float ('rgb_dr', 1.0, 'rgb dropout ratio')
tf.app.flags.DEFINE_float ('flow_dr', 1.0, 'flow dropout ratio')
tf.app.flags.DEFINE_float ('fusion_dr',1.0,'fusion dropout ratio')
tf.app.flags.DEFINE_integer ('num_segments', 3, 'num segments for TSN')
tf.app.flags.DEFINE_boolean ('use_pbn', False, 'use partial batch norm')
tf.app.flags.DEFINE_boolean ('use_batch_norm', False, 'use fusion batch norm')
tf.app.flags.DEFINE_string ('model_type', 'resnet50', 'use which model vgg,inception,resnet and so on')
tf.app.flags.DEFINE_integer ('eval_freq',5,'eval frequence')
tf.app.flags.DEFINE_float ('clip_grad', 20, 'clip gradient norm')
tf.app.flags.DEFINE_string ('fusion_mode', 'add', 'use add or concat to fusion rgb and flow feature')


Flags = tf.app.flags.FLAGS
tf.logging.set_verbosity (tf.logging.INFO)

rgb_dr = Flags.rgb_dr
flow_dr = Flags.flow_dr
fusion_dr = Flags.fusion_dr
split = Flags.split
use_pbn = Flags.use_pbn
num_segments = Flags.num_segments
eval_type = Flags.eval_type
lr_step = eval (Flags.lr_step)
gpu_nums = Flags.gpu_nums
model_type = Flags.model_type
fusion_type = Flags.fusion_type
eval_freq = Flags.eval_freq
clip_grad = Flags.clip_grad
fusion_mode = Flags.fusion_mode
use_batch_norm = Flags.use_batch_norm


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
        

        # init learning rate
        self.lr = tf.placeholder (tf.float32, [])
        # init label
        self.y_ = tf.placeholder (tf.float32, [None, self._NUM_CLASSES])
        # init optimizer
        self.opt = tf.train.MomentumOptimizer (self.lr, 0.9)
        self.is_training = tf.placeholder(tf.bool)

    def init_dataset(self):
        dataset = Flags.dataset
        if dataset == 'UCF101':
            self.dataset = ucf_ts.ucf_dataset (split_number=split, is_training_split=True,
                                                      batch_size=self.batch_size, epoch=1,
                                                      eval_type=eval_type, frame_counts=self._FRAME_COUNTS,
                                                      image_size=self._IMAGE_SIZE,
                                                      prefetch_buffer_size=self.batch_size).dataset ()
            self.val_dataset = ucf_ts_val.ucf_dataset (split_number=split, is_training_split=False,
                                                          batch_size=self.batch_size, epoch=1,
                                                          eval_type=eval_type,frame_counts=self._FRAME_COUNTS,
                                                          image_size=self._IMAGE_SIZE,
                                                          prefetch_buffer_size=self.batch_size).dataset ()
        elif dataset == 'hmdb51':
            self.dataset = hmdb_video_2d.hmdb_dataset (split_number=split, is_training_split=True,
                                                       batch_size=self.batch_size, epoch=1,
                                                       eval_type=eval_type,
                                                       image_size=self._IMAGE_SIZE,
                                                       prefetch_buffer_size=self.batch_size).dataset ()
            self.val_dataset = hmdb_video_2d.hmdb_dataset (split_number=split, is_training_split=False,
                                                           batch_size=self.batch_size, epoch=1,
                                                           eval_type=eval_type,
                                                           image_size=self._IMAGE_SIZE,
                                                           prefetch_buffer_size=self.batch_size).dataset ()

        iter = tf.data.Iterator.from_structure (self.dataset.output_types, self.dataset.output_shapes)
        self.next_element = iter.get_next ()
        self.training_init_op = iter.make_initializer (self.dataset)
        self.validation_init_op = iter.make_initializer (self.val_dataset)

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
                        self.flow_fusion_model = ts_non_local_net.FeatureHierachyNetwork(num_classes=self._NUM_CLASSES,fusion_mode = fusion_mode,use_batch_norm = use_batch_norm)
        
        

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
            rgb_fusion_logits, endpoints = self.rgb_fusion_model (rgb_endpoints,is_training=self.is_training,
                                                            dropout_keep_prob=fusion_dr)

        if eval_type in ['flow', 'joint'] and flow_input is not None:
            flow_logits, flow_endpoints = self.flow_model (flow_input, is_training=False,
                                                        dropout_keep_prob=1.0)
            flow_fusion_logits, endpoints = self.flow_fusion_model (flow_endpoints,is_training=self.is_training,
                                                            dropout_keep_prob=fusion_dr)

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
            model_logits = self.train_model (self.rgb_train,self.flow_train)
        else:
            model_logits = self.Multigpu_train(self.train_model,gpu_nums,self.rgb_train,self.flow_train)

        self.get_saver()

        model_predictions = tf.nn.softmax (model_logits)
        self.cross_entropy = tf.reduce_mean (-tf.reduce_sum (self.y_ * tf.log (model_predictions + 1e-10),
                                                             reduction_indices=[1]))
        
        self.total_loss = self.cross_entropy + self.regular_loss
        tf.summary.scalar ('loss', tensor=self.cross_entropy)
        tf.summary.scalar (name='train_regular_loss', tensor=self.regular_loss)
        self.global_step = tf.Variable (0, trainable=False)
        update_ops = tf.get_collection (tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies (update_ops):
            if eval_type == 'joint':
                self.train_op = self.opt.minimize (self.total_loss, global_step=self.global_step,colocate_gradients_with_ops=True)
            else:
                self.train_op = self.opt.minimize (self.total_loss,var_list = self.fusion_var_map, global_step=self.global_step,colocate_gradients_with_ops=True)
            
        correct_prediction = tf.equal (tf.argmax (model_predictions, 1), tf.argmax (self.y_, 1))
        self.train_accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))
        tf.summary.scalar (name='train_accuarcy', tensor=self.train_accuracy)

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
        self.best_val_accuracy = 0

        self.best_saver_path = os.path.join ('/mnt/zhujian/action_recognition/3d/', self.dataset_name,
                                             str (split), self.base_net, eval_type,fusion_type,fusion_mode)
        if os.path.exists (self.best_saver_path) is False:
            os.makedirs (self.best_saver_path)
        

    def _restore_model(self):
        if self.reboot:

                path = os.path.join ('/mnt/zhujian/action_recognition/two_stream/', self.dataset_name,
                                           str (split), self.base_net,eval_type,"final")
                ckpt = tf.train.get_checkpoint_state (path)
                if ckpt is not None:
                    if eval_type in ['rgb','joint']:
                        self.rgb_saver.restore (self.sess, ckpt.model_checkpoint_path)
                    if eval_type in ['flow','joint']:
                        self.flow_saver.restore (self.sess, ckpt.model_checkpoint_path)
        else:
            path = os.path.join ('/mnt/zhujian/action_recognition/3d/', self.dataset_name, str (split),
                                        self.base_net,eval_type,fusion_type,fusion_mode)
            ckpt = tf.train.get_checkpoint_state (path)
            if ckpt is not None:
                self.saver.restore (self.sess, ckpt.model_checkpoint_path)

    def one_epoch_train(self, epoch):
        self.sess.run (self.training_init_op)
        one_epoch_loss = []
        one_epoch_accuracy = []
        feed_dict = {}
        while 1:
            try:
                read_time = time.time ()

                if eval_type == 'rgb':
                    rgb_file, label = self.sess.run (self.next_element)
                elif eval_type == 'flow':
                    flow_file, label = self.sess.run (self.next_element)
                else:
                    rgb_file, flow_file, label = self.sess.run (self.next_element)
                duration = time.time () - read_time
                if eval_type in ['rgb', 'joint']:
                    feed_dict[self.rgb_train] = rgb_file
                if eval_type in ['flow', 'joint']:
                    feed_dict[self.flow_train] = flow_file

                feed_dict[self.y_] = label
                feed_dict[self.lr] = self.learning_rate
                feed_dict[self.is_training] = True
                start_time = time.time ()

                step, summary ,loss, _, out_predictions, r_loss = self.sess.run (
                    [self.global_step, self.merge_summary,self.cross_entropy, self.train_op, self.train_accuracy,
                     self.regular_loss], feed_dict=feed_dict)
                train_time = time.time () - start_time
                one_epoch_loss.append (loss)
                one_epoch_accuracy.append (out_predictions)
                print (
                    'the training step is %d , the out prediction is %.4f,duratio is %.3f,'
                    'train time is %.2f,the train loss is %.4f,the regular loss is %.4f'
                    %
                    (step, out_predictions, duration, train_time, loss, r_loss))
                self.train_writer.add_summary (summary, step)

                if eval_type == 'rgb':
                    del rgb_file,label
                elif eval_type == 'flow':
                    del flow_file,label
                else:
                    del rgb_file,flow_file,label

            except tf.errors.OutOfRangeError:
                if epoch in lr_step:
                    self.learning_rate /= 10
                train_loss = np.mean (one_epoch_loss)
                train_accuracy = np.mean (one_epoch_accuracy)
                log_file = os.path.join (self.train_log_path, 'train_log.txt')
                with open (log_file, 'a') as f:
                    f.writelines ('At %d epoch , loss is %.3f , accuracy is %.3f\n' %
                                  (epoch, train_loss, train_accuracy))
                break

    def one_epoch_eval(self, epoch):
        self.sess.run (self.validation_init_op)
        valid_accuracy = []
        valid_loss = []
        feed_dict = {}

        while 1:
            try:
                t = time.time ()
                if eval_type == 'rgb':
                    test_rgb, test_label = self.sess.run (self.next_element)
                    feed_dict[self.rgb_train] = test_rgb
                elif eval_type == 'flow':
                    test_flow, test_label = self.sess.run (self.next_element)
                    feed_dict[self.flow_train] = test_flow
                else:
                    test_rgb, test_flow, test_label = self.sess.run (self.next_element)
                    feed_dict[self.flow_train] = test_flow
                    feed_dict[self.rgb_train] = test_rgb

                feed_dict[self.y_] = test_label
                feed_dict[self.lr] = self.learning_rate
                feed_dict[self.is_training] = False
                test_accuracy, test_loss = self.sess.run ([self.train_accuracy, self.cross_entropy],
                                                            feed_dict=feed_dict)
                valid_loss.append (test_loss)
                valid_accuracy.append (test_accuracy)
                print ('epoch %d,the test accuracy is %f,the test loss is %.4f ,use time %.2f ' %
                       (epoch, np.mean (np.array (valid_accuracy)),
                        np.mean (np.array (valid_loss)), time.time () - t))
            except tf.errors.OutOfRangeError:
                log_file = os.path.join (self.train_log_path, 'val_log.txt')
                with open (log_file, 'a') as f:
                    f.writelines ('At %d epoch,the test accuracy is %f,the test loss is %.4f\n' %
                                  (epoch,np.mean (np.array (valid_accuracy)),
                                   np.mean (np.array (valid_loss))))
                try:
                    self.best_saver.save (self.sess, os.path.join (self.best_saver_path, 'final/model.ckpt'))
                except:
                    print('saver last ckpt failed')
                if np.mean (np.array (valid_accuracy)) >= self.best_val_accuracy:
                    self.best_val_accuracy = np.mean (np.array (valid_accuracy))
                    self.best_saver.save (self.sess, os.path.join (self.best_saver_path, 'model.ckpt'))
                break

    def main(self):

        for i in range (self.epoch):
            self.one_epoch_train (i)
            if (i+1) % eval_freq == 0 or i == self.epoch - 1:
                self.one_epoch_eval (i)



def main(argv):
    m = model (video_dir=Flags.video_dir,
               image_size=Flags.image_size,
               num_classes=Flags.num_classes,
               frame_counts=Flags.frame_counts,
               batch_size=Flags.batch_size,
               learning_rate=Flags.learning_rate,
               epoch=Flags.epoch,
               dataset=Flags.dataset,
               reboot=Flags.reboot)
    m.main ()


if __name__ == '__main__':
    tf.app.run (main)
