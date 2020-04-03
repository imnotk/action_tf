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
                 flow_u_path=r'./ucf_flow_video/u',
                 flow_v_path=r'./ucf_flow_video/v',
                 flow_path = r'./ucf101_flow_video/',
                 split_number=0,
                 is_training_split=True,
                 num_segments = 3,
                 frame_counts = 16,
                 image_size=224,
                 batch_size = 12,
                 epoch = 40,
                 num_threads = 8,
                 prefetch_buffer_size = 1000,
                 eval_type='rgb'):
        self._videl_path_include_label = video_path_include_label
        self._flow_path =flow_path
        self._flow_u_path = flow_u_path
        self._flow_v_path = flow_v_path
        self._split_number = split_number
        self._num_segments = num_segments
        self._frame_counts = frame_counts
        self._IMAGE_SIZE = image_size
        self._is_training_split = is_training_split
        self._train_split, self._test_split = self.genrate_data ()
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._batch_size = batch_size
        self._epoch = epoch
        self._prefetch_buffer_size = prefetch_buffer_size
        self._eval_type = eval_type
        if self._is_training_split:
            print('use split %d' % split_number)
            self._path = np.array (self._train_split[split_number][0])
            self._label = self._train_split[split_number][1]
        else:
            self._path = np.array (self._test_split[split_number][0])
            self._label = self._test_split[split_number][1]
        self._num_example = len (self._path)
        self._image_map = {}
        self.num_threads = num_threads
        self.global_set()

    def global_set(self):
        global  _IMAGE_SIZE
        # global  _batch_size
        # global  _eval_type
    
        global _is_training
        _IMAGE_SIZE = self._IMAGE_SIZE
        # _batch_size = self._FRAME_COUNTS
        _num_segments = self._num_segments
        _is_training = self._is_training_split
        # _eval_type = self._eval_type

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
                path = os.path.join (self._videl_path_include_label, trainlist[i][0])
                label_list.append( int(trainlist[i][1]) - 1)
                all_path.append (path)
            train_split[index] = (all_path, label_list)
            index += 1
        index = 0
        for testlist in [testlist1, testlist2, testlist3]:
            label_list = []
            all_path = []
            for i in range (testlist.shape[0]):
                path = os.path.join (self._videl_path_include_label, testlist[i])
                i_class = testlist[i].split ('/')[0]
                # labe[class_map[i_class] - 1] = 1
                label = class_map[i_class] - 1
                label_list.append (label)
                testlist[i] = path
                all_path.append (path)
            test_split[index] = (all_path, label_list)
            index += 1
        return train_split, test_split

    def dataset(self):
        rgb_path = list(self._path)
        flow_path = [os.path.join (self._flow_path, path.split ('/')[-2], path.split ('/')[-1]) for
                       path in self._path]
        label = list(self._label)
        # print(type(rgb_path),type(flow_path),type(label),label)
        rgb_dataset = tf.data.Dataset.from_tensor_slices((rgb_path))
        flow_dataset = tf.data.Dataset.from_tensor_slices((flow_path))
        label_dataset = tf.data.Dataset.from_tensor_slices((label))
        if self._eval_type == 'rgb':
            dataset = tf.data.Dataset.zip((rgb_dataset,label_dataset))
        elif self._eval_type == 'flow':
            dataset = tf.data.Dataset.zip((flow_dataset,label_dataset))
        else:
            dataset = tf.data.Dataset.zip((rgb_dataset,flow_dataset,label_dataset))
        print('dataset create successfully')
        dataset = dataset.shuffle(buffer_size=self._num_example,reshuffle_each_iteration=True)
        dataset = dataset.repeat(self._epoch)
        if self._eval_type == 'rgb':
            dataset = dataset.map (
                lambda r_p,  l: tf.py_func (self._py_func_rgb_vp, [r_p, l], [ tf.float32, tf.float32]),
                num_parallel_calls=self.num_threads)
        elif self._eval_type == 'flow':
            dataset = dataset.map (
                lambda f_p, l: tf.py_func (self._py_func_flow_vp, [f_p, l], [tf.float32, tf.float32]),
                num_parallel_calls=self.num_threads)
        else:
            dataset = dataset.map (
                lambda r_p, f_p, l: tf.py_func (self._py_func_vp, [r_p, f_p, l], [tf.float32, tf.float32, tf.float32]),
                num_parallel_calls=self.num_threads)
        print('dataset transformation successfully')
        # dataset = dataset.batch(batch_size=self._batch_size)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=self._batch_size))
        dataset = dataset.prefetch(buffer_size=self._prefetch_buffer_size)
        return dataset

    def val_dataset(self):
        rgb_path = list(self._path)
        flow_path = [os.path.join (self._flow_path, path.split ('/')[-2], path.split ('/')[-1]) for
                       path in self._path]
        label = list(self._label)
        # print(type(rgb_path),type(flow_path),type(label),label)
        rgb_dataset = tf.data.Dataset.from_tensor_slices((rgb_path))
        flow_dataset = tf.data.Dataset.from_tensor_slices((flow_path))
        label_dataset = tf.data.Dataset.from_tensor_slices((label))
        if self._eval_type == 'rgb':
            dataset = tf.data.Dataset.zip((rgb_dataset,label_dataset))
        elif self._eval_type == 'flow':
            dataset = tf.data.Dataset.zip((flow_dataset,label_dataset))
        else:
            dataset = tf.data.Dataset.zip((rgb_dataset,flow_dataset,label_dataset))
        print('dataset create successfully')
        # dataset = dataset.shuffle(buffer_size=self._num_example,reshuffle_each_iteration=True)
        dataset = dataset.repeat(self._epoch)
        if self._eval_type == 'rgb':
            dataset = dataset.map (
                lambda r_p,  l: tf.py_func (self._py_func_rgb_val_vp, [r_p, l], [ tf.float32, tf.float32]),
                num_parallel_calls=self.num_threads)
        elif self._eval_type == 'flow':
            dataset = dataset.map (
                lambda f_p, l: tf.py_func (self._py_func_flow_val_vp, [f_p, l], [tf.float32, tf.float32]),
                num_parallel_calls=self.num_threads)
        else:
            dataset = dataset.map (
                lambda r_p, f_p, l: tf.py_func (self._py_func_val_vp, [r_p, f_p, l], [tf.float32, tf.float32, tf.float32]),
                num_parallel_calls=self.num_threads)
        print('dataset transformation successfully')
        # dataset = dataset.batch(batch_size=self._batch_size)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=self._batch_size))
        dataset = dataset.prefetch(buffer_size=self._prefetch_buffer_size)
        return dataset


    @staticmethod
    def _py_func_rgb_vp(rgb_path,  label):

        rgb_path = rgb_path.decode ()

        num_segments = 3
        # rgb_file = _rgb_vp (rgb_path)
        rgb_cap = cv2.VideoCapture(rgb_path)
        rgb_len = rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        segments_list = np.array_split(np.arange(0,rgb_len),num_segments)
        image_list = []
        while num_segments:
            index = np.random.choice(segments_list[num_segments-1])
            rgb_cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            flag , image = rgb_cap.read()
            if image is None:
                continue 
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = np.float32(image)
            image = cv2.resize(image,(340,256))
            image = DataAugmentation.Multiscale_crop(image)
            image = DataAugmentation.horizontal_flip(image)
            image = cv2.resize(image,(224,224))
            image = normalize(image)
            image_list.append(image)
            num_segments -= 1

        
        assert len(image_list) != num_segments

        rgb_cap.release()
        rgb_file = image_list

        if label is not None:
            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return rgb_file,  one_hot_label
        
        return rgb_file

    @staticmethod
    def _py_func_flow_vp(flow_path, label):

        flow_path = flow_path.decode ()

        # flow_file = _flow_vp (flow_path)
        flow_cap = cv2.VideoCapture(flow_path)
        flow_len = flow_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        num_segments = 3
        segments_list = np.array_split(np.arange(0,flow_len),num_segments)

        flag = True
        image_list = []
        while num_segments:
            index = np.random.choice(segments_list[num_segments-1])
            flow_cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            img_list = []
            for i in range(10):
                f , img = flow_cap.read()
                if img is not None:
                    img_list.append(img[...,:2])
            if len(img_list) != 10:
                continue

            image = np.concatenate(img_list,axis=-1)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = np.float32(image)
            image = cv2.resize(image,(340,256))
            image = DataAugmentation.Multiscale_crop(image,is_flow=True)
            image = DataAugmentation.horizontal_flip(image)
            image = cv2.resize(image,(224,224))
            image = (image / 255) * 2 - 1
            image_list.append(image)
            num_segments -= 1

        flow_file = image_list


        flow_cap.release()
        if label is not None:
            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return flow_file, one_hot_label
        
        return flow_file
            

    @staticmethod
    def _py_func_vp(rgb_path, flow_path, label):

        #############
        # print('rgb_path begin to read')
        # rgb_path = rgb_path.decode ()
        # flow_path = flow_path.decode ()
        # rgb_file, flow_file = _vp ([rgb_path, flow_path])
        rgb_file = ucf_dataset._py_func_rgb_vp(rgb_path,label=None)
        flow_file = ucf_dataset._py_func_flow_vp(flow_path,label=None)

        one_hot_label = np.zeros (101, dtype=np.float32)
        one_hot_label[label] = 1

        return rgb_file, flow_file, one_hot_label

    @staticmethod
    def _py_func_rgb_val_vp(rgb_path, label):

        rgb_path = rgb_path.decode ()

        num_segments = 3
        # rgb_file = _rgb_vp (rgb_path)
        rgb_cap = cv2.VideoCapture (rgb_path)
        rgb_len = rgb_cap.get (cv2.CAP_PROP_FRAME_COUNT)
        segments_list = np.array_split (np.arange (0, rgb_len), num_segments)
        image_list = []
        while num_segments:
            index = np.random.choice (segments_list[num_segments - 1])
            rgb_cap.set (cv2.CAP_PROP_POS_FRAMES, index)
            flag, image = rgb_cap.read ()
            if image is None:
                continue
            image = np.float32 (image)
            image = cv2.resize (image, (340, 256))
            image = DataAugmentation.center_crop (image,224,224)
            image = normalize (image)
            image_list.append (image)
            num_segments -= 1

        assert len (image_list) != num_segments

        rgb_cap.release ()
        rgb_file = image_list

        if label is not None:
            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return rgb_file, one_hot_label

        return rgb_file

    @staticmethod
    def _py_func_flow_val_vp(flow_path, label):

        flow_path = flow_path.decode ()

        # flow_file = _flow_vp (flow_path)
        flow_cap = cv2.VideoCapture (flow_path)
        flow_len = flow_cap.get (cv2.CAP_PROP_FRAME_COUNT)
        num_segments = 3
        segments_list = np.array_split (np.arange (0, flow_len), num_segments)

        flag = True
        image_list = []
        while num_segments:
            index = np.random.choice (segments_list[num_segments - 1])
            flow_cap.set (cv2.CAP_PROP_POS_FRAMES, index)
            img_list = []
            for i in range (10):
                f, img = flow_cap.read ()
                if img is not None:
                    img_list.append (img[..., :2])
            if len (img_list) != 10:
                continue

            image = np.concatenate (img_list, axis=-1)
            image = np.float32 (image)
            image = cv2.resize (image, (340, 256))
            image = DataAugmentation.center_crop (image,224,224)
            image = cv2.resize (image, (224, 224))
            image = (image / 255) * 2 - 1
            image_list.append (image)
            num_segments -= 1

        flow_file = image_list

        flow_cap.release ()
        if label is not None:
            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return flow_file, one_hot_label

        return flow_file

    @staticmethod
    def _py_func_val_vp(rgb_path, flow_path, label):

        #############
        # print('rgb_path begin to read')
        rgb_file = ucf_dataset._py_func_rgb_val_vp(rgb_path,label=None)
        flow_file = ucf_dataset._py_func_flow_val_vp(flow_path,label=None)

        one_hot_label = np.zeros (101, dtype=np.float32)
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


if __name__ == '__main__':
    from tensorflow.contrib.slim.nets import resnet_v1
    import time
    import tensorflow as tf
    tf.enable_eager_execution()

    m_d = ucf_dataset (split_number=0, is_training_split=False,
                                    batch_size=20, epoch=1,
                                    frame_counts=25, eval_type='rgb',
                                    image_size=224,
                                    prefetch_buffer_size=20).val_dataset ()
    test_m_d = ucf_dataset (split_number=0, is_training_split=False,
                                         batch_size=20, epoch=1,
                                         frame_counts=10, eval_type='rgb',
                                         image_size=224,
                                         prefetch_buffer_size=20).val_dataset ()
    iter = m_d.make_one_shot_iterator()
    for i in range(5):
        t = time.time()
        g = iter.next()
        # r = np.squeeze(r)
        end_t = time.time() - t
        print(g[0].shape,end_t)
    