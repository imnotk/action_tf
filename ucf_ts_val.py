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
import gc

class ucf_dataset:

    train_test_list = {
        'test1': 'UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist01.txt',
        'test2': 'UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist02.txt',
        'test3': 'UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist03.txt',
        'train1': 'UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist01.txt',
        'train2': 'UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist02.txt',
        'train3': 'UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist03.txt',

    }
    classind = 'UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt'
    video_path = 'UCF-101'

    def __init__(self,
                 video_path_include_label='./UCF-101/',
                 flow_u_path = 'ucf101_tvl1_flow/tvl1_flow/u',
                 flow_v_path = 'ucf101_tvl1_flow/tvl1_flow/v',
                 split_number=0,
                 is_training_split=True,
                 frame_counts=10,
                 image_size=224,
                 batch_size = 24,
                 epoch=40,
                 new_length = 10,
                 prefetch_buffer_size=24,
                 eval_type='rgb'):
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
        self._new_length = new_length
        self._eval_type = eval_type
        self._prefetch_buffer_size = prefetch_buffer_size
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
        global  _frame_counts
        global  _new_length
        _IMAGE_SIZE = self._IMAGE_SIZE
        _batch_size = self._batch_size
        _frame_counts = self._FRAME_COUNTS
        _new_length = self._new_length

    def genrate_data(self):
        trainlist1 = np.genfromtxt (self.train_test_list['train1'], dtype='U')
        trainlist2 = np.genfromtxt (self.train_test_list['train2'], dtype='U')
        trainlist3 = np.genfromtxt (self.train_test_list['train3'], dtype='U')
        testlist1 = np.genfromtxt (self.train_test_list['test1'], dtype='U')
        testlist2 = np.genfromtxt (self.train_test_list['test2'], dtype='U')
        testlist3 = np.genfromtxt (self.train_test_list['test3'], dtype='U')
        classind = np.genfromtxt (self.classind, dtype='U')
        class_map = {}
        for i in classind:
            class_map[i[1]] = int (i[0])
        train_split = {}
        test_split = {}
        index = 0
        for trainlist in [trainlist1, trainlist2, trainlist3]:
            label_list = []
            all_path = []
            for i in range (trainlist.shape[0]):
                path = trainlist[i][0]
                label_list.append( int(trainlist[i][1]) - 1)
                all_path.append (path)
            train_split[index] = (all_path, label_list)
            index += 1
        index = 0
        for testlist in [testlist1, testlist2, testlist3]:
            label_list = []
            all_path = []
            for i in range (testlist.shape[0]):
                path = testlist[i]
                i_class = testlist[i].split ('/')[0]
                label = class_map[i_class] - 1
                label_list.append (label)
                testlist[i] = path
                all_path.append (path)
            test_split[index] = (all_path, label_list)
            index += 1
        return train_split, test_split


    
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
    def pick_index(length,counts,is_temporal_segment=True):
        if is_temporal_segment:
            k = random.random()
            if k < 0:
                index = np.random.randint(0,length-counts)
                return np.range(index,index+counts)
            else:
                g = np.array_split(np.arange(length),counts)
                index_list = []
                for i in g:
                    index_list.append(np.random.choice(i))
                return index_list
        else:
            index = np.random.randint(0,length-counts)
            return np.range(index,index+counts)

    @staticmethod
    def _py_func_rgb_vp(rgb_path, label):

        rgb_path = rgb_path.decode ()

        rgb_cap = cv2.VideoCapture (rgb_path)
        rgb_len = rgb_cap.get (cv2.CAP_PROP_FRAME_COUNT)

        image_list = []
        
        index_list = ucf_dataset.pick_index(rgb_len,_frame_counts)

        for index in index_list:
            rgb_cap.set (cv2.CAP_PROP_POS_FRAMES, index)
            flag, image = rgb_cap.read ()
            if image is not None:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = cv2.resize (image, (340, 256))
                image_list.append(image)

        if len(image_list) != _frame_counts:
            image_list = np.concatenate([image_list] * _frame_counts,axis=0)[:_frame_counts]   
            
        assert image_list != []

        rgb_cap.release ()
        image = np.stack(image_list,axis=0)
        rgb_file =  DataAugmentation.center_crop(image,224,224)
        rgb_file = np.float32([cv2.resize(x,(224,224)) for x in rgb_file])

        rgb_file = np.float32(rgb_file)
        rgb_file = normalize (rgb_file)

        assert rgb_file is not None

        if label is not None:
            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return rgb_file, one_hot_label

        return rgb_file

    @staticmethod
    def _py_func_flow_vp(flow_path, label):

        f_upath , f_vpath = flow_path
        flow_u_path = f_upath.decode ()
        flow_v_path = f_vpath.decode ()
        flow_file = os.listdir (flow_u_path)
        flow_file = sorted(flow_file)
        flow_len = len(flow_file)

        index_list = ucf_dataset.pick_index(flow_len - _new_length,_frame_counts)

        flow_img = []
        for j in index_list:
            img_list = []
            for i in range(j,j + _new_length):
                img_u_path = os.path.join(flow_u_path,flow_file[i])
                img_v_path = os.path.join(flow_v_path,flow_file[i])
                
                img_u = cv2.imread(img_u_path,0)
                img_v = cv2.imread(img_v_path,0)
                if img_u is None or img_v is None:
                    continue
                img = np.stack([img_u,img_v],axis=-1)
                img_list.append(img)

            img = np.concatenate(img_list,axis=-1)
            img = cv2.resize(img,(340,256))
            img = DataAugmentation.center_crop(img,224,224)
            img = cv2.resize(img,(224,224))
            img = np.float32(img)
            img = (img / 255 - 0.5) / 0.226
            flow_img.append(img)

        if label is not None:

            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return flow_img, one_hot_label

        return flow_img

    @staticmethod
    def _py_func_vp(rgb_path, flow_path, label):

        #############
        rgb_file = ucf_dataset._py_func_rgb_vp (rgb_path, label=None)
        flow_file = ucf_dataset._py_func_flow_vp (flow_path, label=None)

        one_hot_label = np.zeros (101, dtype=np.float32)
        one_hot_label[label] = 1

        return rgb_file, flow_file, one_hot_label



def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = img / 255
    img_channel = img.shape[-1]
    mean = mean * (img_channel // len (mean))
    std = std * (img_channel // len (std))
    for i in range (img_channel):
        img[..., i] = (img[..., i] - mean[i]) / std[i]
    return img

if __name__ == '__main__':
    import tensorflow as tf
    tf.enable_eager_execution()
    m = ucf_dataset(batch_size=1,eval_type='flow').dataset()
    iter = m.make_one_shot_iterator()
    g = iter.next()
    print(g[0].shape)