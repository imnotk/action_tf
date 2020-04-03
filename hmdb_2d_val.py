from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import tensorflow as tf
import skvideo.io
import cv2
import numpy as np
from Augamentation import DataAugmentation
import random



class hmdb_dataset:

    def __init__(self,
                 video_path_include_label='./hmdb51/',
                 flow_u_path = 'hmdb51_tvl1_flow/u',
                 flow_v_path = 'hmdb51_tvl1_flow/v',
                 split_number=0,
                 is_training_split=True,
                 frame_counts=10,
                 image_size=224,
                 batch_size = 24,
                 epoch=40,
                 new_length = 10,
                 prefetch_buffer_size=24,
                 eval_type='rgb',
                 test_crop='multi',
                 preprocess_name='pytorch'):
        self._videl_path_include_label = video_path_include_label
        self._flow_u_path = flow_u_path
        self._flow_v_path = flow_v_path
        self._split_number = split_number
        self._FRAME_COUNTS = frame_counts
        self._IMAGE_SIZE = image_size
        self._is_training_split = is_training_split
        self._train_split, self._test_split = self.genrate_data ()
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._batch_size = batch_size
        self._epoch = epoch
        self._test_crop = test_crop
        self._eval_type = eval_type
        self._prefetch_buffer_size = prefetch_buffer_size
        self._preprocess_name = preprocess_name
        self._new_length = new_length
        if self._is_training_split:
            print('use training split %d' % split_number)
            self._path = np.array (self._train_split[split_number][0])
            self._label = self._train_split[split_number][1]
        else:
            print('use test split %d' % split_number)
            self._path = np.array (self._test_split[split_number][0])
            self._label = self._test_split[split_number][1]
        self._num_example = len (self._path)
        self._image_map = {}
        self.global_set()

    def global_set(self):
        global  _IMAGE_SIZE
        global  _batch_size
        global  test_crop
        global _preprocess_name 
        global _new_length
        _IMAGE_SIZE = self._IMAGE_SIZE
        _batch_size = self._batch_size
        _frame_counts = self._FRAME_COUNTS
        test_crop = self._test_crop
        _preprocess_name = self._preprocess_name
        _new_length = self._new_length

    def genrate_data(self,shuffle = True,to_one_hot = False):

        hmdb_train_test_path = 'test_train_splits/testTrainMulti_7030_splits/'
        path_list = os.listdir(hmdb_train_test_path)
        hmdb_path = './hmdb51/'
        test_split = {}
        train_split = {}
        hmdb_label = {}
        hmdb_class = {}
        for i,name in enumerate(os.listdir('./hmdb51')):
            hmdb_label[name] = i
            hmdb_class[i] = name
        self._class = hmdb_label
        self._class_ind = hmdb_class
        test_path_1 = [];train_path_1 = []
        test_path_2 = [];train_path_2 = []
        test_path_3 = [];train_path_3 = []
        test_label_1 = [];train_label_1 = []
        test_label_2 = [];train_label_2 = []
        test_label_3 = [];train_label_3 = []
        for i,name in enumerate(path_list):
            split_index = int(name[-5]) - 1
            name_path = hmdb_train_test_path + name
            with open(name_path,'r') as f:
                name_label = hmdb_label[name[:-16]]
                name_list = f.readlines()
                for t_list in name_list:
                    t_path = hmdb_class[name_label]+'/'+t_list.split()[0]
                    # n_label = np.zeros([1, 51], dtype=np.float32)
                    # n_label[0][name_label] = 1
                    if t_list.split()[1] == '1':
                        if split_index == 0:
                            train_path_1.append(t_path);train_label_1.append(name_label)
                        elif split_index == 1:
                            train_path_2.append(t_path);train_label_2.append(name_label)
                        elif split_index == 2:
                            train_path_3.append(t_path);train_label_3.append(name_label)
                    elif t_list.split()[1] == '2':
                        if split_index == 0:
                            test_path_1.append(t_path);test_label_1.append(name_label)
                        elif split_index == 1:
                            test_path_2.append(t_path);test_label_2.append(name_label)
                        elif split_index == 2:
                            test_path_3.append(t_path);test_label_3.append(name_label)
        train_split[0] = (train_path_1,train_label_1)
        train_split[1] = (train_path_2,train_label_2)
        train_split[2] = (train_path_3,train_label_3)
        test_split[0] = (test_path_1,test_label_1)
        test_split[1] = (test_path_2,test_label_2)
        test_split[2] = (test_path_3,test_label_3)
        return train_split,test_split

    def dataset(self):
        rgb_path = [os.path.join(self._videl_path_include_label,path) for path in self._path]
        flow_u_path = [os.path.join (self._flow_u_path, path[:-4]) for path in self._path]
        flow_v_path = [os.path.join (self._flow_v_path, path[:-4]) for path in self._path]
        label = list(self._label)

        rgb_dataset = tf.data.Dataset.from_tensor_slices((rgb_path))
        flow_dataset = tf.data.Dataset.from_tensor_slices ((flow_u_path,flow_v_path))
        label_dataset = tf.data.Dataset.from_tensor_slices((label))
        if self._eval_type == 'rgb':
            dataset = tf.data.Dataset.zip((rgb_dataset,label_dataset))
        elif self._eval_type == 'flow':
            dataset = tf.data.Dataset.zip ((flow_dataset,label_dataset))
        else:
            dataset = tf.data.Dataset.zip((rgb_dataset,flow_dataset,label_dataset))
        print('dataset create successfully')
        dataset = dataset.shuffle(buffer_size=self._num_example,reshuffle_each_iteration=True)
        dataset = dataset.repeat(self._epoch)
        if self._eval_type == 'rgb':
            dataset = dataset.map (
                lambda r_p,  l: tf.py_func (self._py_func_rgb_vp, [r_p, l], [ tf.float32, tf.float32]),
                num_parallel_calls=os.cpu_count())
        elif self._eval_type == 'flow':
            dataset = dataset.map (
                lambda f_p, l: tf.py_func (self._py_func_flow_vp, [f_p, l], [tf.float32, tf.float32]),
                num_parallel_calls=os.cpu_count())
        else:
            dataset = dataset.map (
                lambda r_p, f_p, l: tf.py_func (self._py_func_vp, [r_p, f_p, l], [tf.float32, tf.float32, tf.float32]),
                num_parallel_calls=os.cpu_count())
        print('dataset transformation successfully')
        dataset = dataset.batch(batch_size=self._batch_size,drop_remainder=True)
        # dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=self._batch_size))
        dataset = dataset.prefetch(buffer_size=self._prefetch_buffer_size)
        return dataset


    @staticmethod
    def _py_func_rgb_vp(rgb_path,  label):

        rgb_path = rgb_path.decode ()

        rgb_cap = cv2.VideoCapture(rgb_path)
        rgb_len = rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT)

        while 1:
            index = np.random.randint(0,rgb_len)
            rgb_cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            _ , image = rgb_cap.read()
            if image is not None:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = np.float32(image)
                image = cv2.resize(image,(340,256))
                rgb_file = DataAugmentation.center_crop(image,224,224)
                rgb_file = cv2.resize(rgb_file,(224,224))
                if _preprocess_name == 'pytorch':
                    rgb_file = normalize(rgb_file)
                elif _preprocess_name == 'tf':
                    rgb_file = tf_preprocess(rgb_file)
                break

        rgb_cap.release()

        if label is not None:
            one_hot_label = np.zeros (51, dtype=np.float32)
            one_hot_label[label] = 1

            return rgb_file,  one_hot_label
        
        return rgb_file

    @staticmethod
    def _py_func_flow_vp(flow_path, label):
        
        f_upath , f_vpath = flow_path
        flow_u_path = f_upath.decode ()
        flow_v_path = f_vpath.decode ()
        flow_file = os.listdir (flow_u_path)
        flow_file = sorted(flow_file)
        index = np.random.randint(0,len(flow_file)- _new_length)
        img_list = []
        for i in range(index,index+ _new_length):
            img_u_path = os.path.join(flow_u_path,flow_file[i])
            img_v_path = os.path.join(flow_v_path,flow_file[i])
            
            img_u = cv2.imread(img_u_path,0)
            img_v = cv2.imread(img_v_path,0)

            img = np.stack([img_u,img_v],axis=-1)
            img_list.append(img)

        img = np.concatenate(img_list,axis=-1)
        img = cv2.resize(img,(340,256))
        img = DataAugmentation.center_crop(img,224,224)
        img = cv2.resize(img,(224,224))
        img = np.float32(img)
        if _preprocess_name == 'pytorch':
            img = (img / 255 - 0.5) / 0.226
        elif _preprocess_name == 'tf':
            img = tf_preprocess(img)

        if label is not None:

            one_hot_label = np.zeros (51, dtype=np.float32)
            one_hot_label[label] = 1

            return img, one_hot_label

        return img

    @staticmethod
    def _py_func_vp(rgb_path, flow_path, label):

        rgb_file = hmdb_dataset._py_func_rgb_vp (rgb_path,None)
        flow_file = hmdb_dataset._py_func_flow_vp (flow_path,None)

        one_hot_label = np.zeros (51, dtype=np.float32)
        one_hot_label[label] = 1

        return rgb_file, flow_file, one_hot_label

    


def normalize(img,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    img = img/255
    img_channel = img.shape[-1]
    mean = mean * (img_channel // len(mean))
    std = std * (img_channel // len(std))
    for i in range(img_channel):
        img[...,i] = (img[...,i] - mean[i]) / std[i]
    return img

def tf_preprocess(img):
    img = img/255
    img = (img - 0.5) * 2
    return img

def subtract_mean(img,is_rgb=True):
    if is_rgb:
        mean = [123.68,116.78,103.94]
        for i in range(3):
            img[...,i] = img[...,i] - mean[i] 
    else:
        img -= 114.8
    return img

if __name__ == '__main__':
    from tensorflow.contrib.slim.nets import resnet_v1
    import time
    import tensorflow as tf
    tf.enable_eager_execution()

    m_d = hmdb_dataset (split_number=0, is_training_split=True,
                                    batch_size=1, epoch=10,
                                    frame_counts=25, eval_type='rgb',
                                    image_size=224,
                                    prefetch_buffer_size=1).dataset ()
    
    iter = m_d.make_one_shot_iterator()
    for i in range(20):
        t = time.time()
        g = iter.next()
        # r = np.squeeze(r)
        end_t = time.time() - t
        print(g[0].shape,g[1].shape,end_t)