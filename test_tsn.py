from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from model import resnet_tsn
from model import bn_inception_tsn
import os
import time
import numpy as np
import pickle
import ucf_ts,hmdb_ts

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gpu_options = tf.GPUOptions (per_process_gpu_memory_fraction=0.98, allow_growth=True)
file_save_path = '/mnt/zhujian/action_recognition'
log_dir = 'log_dir/'

tf.app.flags.DEFINE_string ('video_dir', './UCF-101/', 'the orginal and optical video directory')
tf.app.flags.DEFINE_integer ('image_size', 224, 'the uniform input video size')
tf.app.flags.DEFINE_integer ('num_classes', 101, 'the classes number of the video dataset')
tf.app.flags.DEFINE_integer ('frame_counts', 25, 'the input video frame counts')
tf.app.flags.DEFINE_integer ('batch_size', 1, 'the inputed batch size ')
tf.app.flags.DEFINE_integer ('gpu_nums', 2, 'the inputed batch size ')
tf.app.flags.DEFINE_string ('dataset', 'UCF101', 'the training dataset')
tf.app.flags.DEFINE_string ('eval_type', 'rgb', 'rgb flow or joint')
tf.app.flags.DEFINE_boolean ('reboot', True, 'reboot traing process if True else False')
tf.app.flags.DEFINE_integer ('split', 0, 'split number ')
tf.app.flags.DEFINE_integer ('num_segments', 3, 'num segments for TSN') 
tf.app.flags.DEFINE_boolean ('use_pbn', False, 'use partial batch norm')
tf.app.flags.DEFINE_string ('model_type', 'resnet50', 'use which model vgg,inception,resnet and so on')
tf.app.flags.DEFINE_integer('video_split',10,'video split')
tf.app.flags.DEFINE_string ('test_crop', 'center', 'use which test_crop center or multiscale')
tf.app.flags.DEFINE_integer ('new_length', 10, 'optical flow length for TSN')

Flags = tf.app.flags.FLAGS
tf.logging.set_verbosity (tf.logging.INFO)

video_split = Flags.video_split
split = Flags.split
use_pbn = Flags.use_pbn
num_segments = Flags.num_segments
eval_type = Flags.eval_type
gpu_nums = Flags.gpu_nums
reboot = Flags.reboot
model_type = Flags.model_type
test_crop = Flags.test_crop
new_length = Flags.new_length
print(model_type)


if test_crop == 'multi':
    factor = 10
else:
    factor = 2


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
                 video_split = 10,
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
        self.video_split = video_split

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
                                             [None,self._FRAME_COUNTS * factor, self._IMAGE_SIZE, self._IMAGE_SIZE, 3])
        else:
            self.rgb_train = None

        if eval_type in ['flow', 'joint']:
            self.flow_train = tf.placeholder (tf.float32,
                                              [None, self._FRAME_COUNTS * factor, self._IMAGE_SIZE, self._IMAGE_SIZE, new_length * 2])
        else:
            self.flow_train = None

        # init label
        self.y_ = tf.placeholder (tf.float32, [None, self._NUM_CLASSES])
       

    def init_dataset(self):
        dataset = Flags.dataset
        if dataset == 'UCF101':
            self.dataset = ucf_ts.ucf_dataset (split_number=split, is_training_split=False,
                                                      batch_size=self.batch_size, epoch=1,test_crop=test_crop,
                                                      eval_type=eval_type,video_split=self.video_split,new_length=new_length,
                                                      image_size=self._IMAGE_SIZE,frame_counts=self._FRAME_COUNTS,
                                                      prefetch_buffer_size=self.batch_size).test_dataset ()
            
        elif dataset == 'hmdb51':
            self.dataset = hmdb_ts.hmdb_dataset (split_number=split, is_training_split=False,
                                                      batch_size=self.batch_size, epoch=1,test_crop=test_crop,
                                                      eval_type=eval_type,video_split=self.video_split,new_length=new_length,
                                                      image_size=self._IMAGE_SIZE,frame_counts=self._FRAME_COUNTS,
                                                      prefetch_buffer_size=self.batch_size).test_dataset ()
           

        iter = tf.data.Iterator.from_structure (self.dataset.output_types, self.dataset.output_shapes)
        self.next_element = iter.get_next ()
        self.training_init_op = iter.make_initializer (self.dataset)

    def init_type(self):
        if model_type == 'resnet50':
            self.base_net = 'resnet50_tsn'
        elif model_type == 'resnet101':
            self.base_net = 'resnet101_tsn'
        elif model_type == 'resnet18':
            self.base_net = 'resnet18_tsn'
        elif model_type == 'bn_inception':
            self.base_net = 'bn_inception_tsn'
            

    def init_model(self):
        if eval_type in ['rgb', 'joint']:
            with tf.variable_scope ('RGB', reuse=tf.AUTO_REUSE):
                if model_type == 'resnet50':
                    self.rgb_model = resnet_tsn.Resnet(num_classes=self._NUM_CLASSES,frame_counts=self._FRAME_COUNTS,final_endpoint='logits',name='resnet_v1_50')
                elif model_type == 'resnet101':
                    self.rgb_model = resnet_tsn.Resnet(num_classes=self._NUM_CLASSES,frame_counts=self._FRAME_COUNTS,final_endpoint='logits',name='resnet_v1_101')
                elif model_type == 'bn_inception':
                    self.rgb_model = bn_inception_tsn.BNInception(num_classes=self._NUM_CLASSES,frame_counts=self._FRAME_COUNTS,final_endpoint='Logits')

        if eval_type in ['flow', 'joint']:
            with tf.variable_scope ('Flow', reuse=tf.AUTO_REUSE):
                if model_type == 'resnet18':
                    self.flow_model = resnet_tsn.Resnet(num_classes=self._NUM_CLASSES,frame_counts=self._FRAME_COUNTS,final_endpoint='logits',name='resnet_v1_18')
                elif model_type == 'resnet50':
                    self.flow_model = resnet_tsn.Resnet(num_classes=self._NUM_CLASSES,frame_counts=self._FRAME_COUNTS,final_endpoint='logits',name='resnet_v1_50')
                elif model_type == 'resnet101':
                    self.flow_model = resnet_tsn.Resnet(num_classes=self._NUM_CLASSES,frame_counts=self._FRAME_COUNTS,final_endpoint='logits',name='resnet_v1_101')
                elif model_type == 'bn_inception':                
                    self.flow_model = bn_inception_tsn.Resnet(num_classes=self._NUM_CLASSES,frame_counts=self._FRAME_COUNTS,final_endpoint='logits',name='resnet_v1_18')
                

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
        try:
            self.rgb_saver = tf.train.Saver(reshape=True,var_list=tf.global_variables(scope='RGB'))
        except:
            print('no rgb modality')
        try:
            self.flow_saver = tf.train.Saver(reshape=True,var_list=tf.global_variables(scope='Flow'))
        except:
            print('no flow modality')
        self.saver = tf.train.Saver (max_to_keep=15, reshape=True)

    def train(self):

        self.model_logits = self.train_model (self.rgb_train,self.flow_train)
        
        self.get_saver()
        self.model_predictions = tf.nn.softmax (self.model_logits)
        correct_prediction = tf.equal (tf.argmax (self.model_predictions, 1), tf.argmax (self.y_, 1))
        self.train_accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))

    def _init_sess(self):
        self.init_op = tf.global_variables_initializer ()
        self.merge_summary = tf.summary.merge_all ()
        self.sess = tf.Session (config=tf.ConfigProto (gpu_options=gpu_options))
        self.sess.run (self.init_op)
        log_path = os.path.join (file_save_path, log_dir, self.dataset_name, str (split), self.base_net,
                                 eval_type)
        if os.path.exists (log_path) is False:
            os.makedirs (log_path)
        self.train_writer = tf.summary.FileWriter (
            log_path, self.sess.graph)

        self.train_log_path = os.path.join (log_dir, self.dataset_name, str (split), self.base_net, eval_type)
        if os.path.exists (self.train_log_path) is False:
            os.makedirs (self.train_log_path)
       
    def _restore_model(self):
            if reboot and eval_type == 'joint':
                rgb_path = os.path.join ('/mnt/zhujian/action_recognition/two_stream/', self.dataset_name, str (split),
                                        self.base_net,'rgb')
                rgb_ckpt = tf.train.get_checkpoint_state(rgb_path)
                if rgb_ckpt is not None:
                    self.rgb_saver.restore(self.sess,rgb_ckpt.model_checkpoint_path)
                    print('rgb reboot successfully')
                flow_path = os.path.join ('/mnt/zhujian/action_recognition/two_stream/', self.dataset_name, str (split),
                                        self.base_net,'flow')
                flow_ckpt = tf.train.get_checkpoint_state(flow_path)
                if flow_ckpt is not None:
                    self.flow_saver.restore(self.sess,flow_ckpt.model_checkpoint_path)
                    print('flow reboot successfully')
            else:        
                path = os.path.join ('/mnt/zhujian/action_recognition/two_stream/', self.dataset_name, str (split),
                                            self.base_net,eval_type)
                ckpt = tf.train.get_checkpoint_state (path)
                print(ckpt)
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
                    test_rgb = np.reshape(test_rgb,[self.batch_size,1,-1,224,224,3])
                elif eval_type == 'flow':
                    test_flow, test_label = self.sess.run (self.next_element)
                    test_flow = np.reshape(test_flow,[self.batch_size,1,-1,224,224,20])
                else:
                    test_rgb, test_flow, test_label = self.sess.run (self.next_element)
                    test_rgb = np.reshape(test_rgb,[self.batch_size,1,-1,224,224,3])
                    test_flow = np.reshape(test_flow,[self.batch_size,1,-1,224,224,20])
                    
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

                logits = np.array(logits)
                for i in range(logits.shape[0]):
                    video_predict.append((test_label[i],logits[i]))

                out_pred = (np.equal (np.argmax (logits, 1), np.argmax (test_label, 1))).sum() / self.batch_size
                test_batch_accuracy.append(out_pred)
                end_t = time.time() - t
                print ('after %d step with %.2fs duratio,the test accuracy is %f,use time %.2fs' % 
                    (k, duratio,np.mean (np.array (test_batch_accuracy)),end_t - duratio))
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
               video_split=Flags.video_split,
               dataset=Flags.dataset,
               reboot=Flags.reboot)
    m.main ()


if __name__ == '__main__':
    tf.app.run (main)
