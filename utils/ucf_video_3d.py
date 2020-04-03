from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import cv2
import numpy as np
from Augamentation import DataAugmentation
import tensorflow as tf
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
                 flow_u_path=r'./ucf_flow_video_xvid/u',
                 flow_v_path=r'./ucf_flow_video_xvid/v',
                 flow_path = r'./ucf101_flow_video_xvid/',
                 split_number=0,
                 is_training_split=True,
                 frame_counts=16,
                 image_size=224,
                 batch_size = 12,
                 video_split = 10,
                 epoch = 40,
                 prefetch_buffer_size = 1000,
                 eval_type='rgb',
                 crop = 'mutiscale'):
        self._videl_path_include_label = video_path_include_label
        self._flow_path =flow_path
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
        self._prefetch_buffer_size = prefetch_buffer_size
        self._eval_type = eval_type
        self._video_split = video_split
        self._crop = crop
        if self._is_training_split:
            self._path = np.array (self._train_split[split_number][0])
            self._label = self._train_split[split_number][1]
        else:
            self._path = np.array (self._test_split[split_number][0])
            self._label = self._test_split[split_number][1]
        self._num_example = len (self._path)
        self._image_map = {}
        self.global_set()

    def global_set(self):
        global  _IMAGE_SIZE
        global  _batch_size
        global  _eval_type
        global _frame_counts
        global _crop
        _IMAGE_SIZE = self._IMAGE_SIZE
        _batch_size = self._video_split
        _is_training = self._is_training_split
        _frame_counts = self._FRAME_COUNTS
        _eval_type = 'rgb'
        _crop = self._crop

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
        train_split = {};
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
        dataset = dataset.shuffle(buffer_size=self._num_example)
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
        # dataset = dataset.batch(batch_size=self._batch_size)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=self._batch_size))
        dataset = dataset.prefetch(buffer_size=self._prefetch_buffer_size)
        return dataset

    def test_dataset(self):
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
        dataset = dataset.repeat(self._epoch)
        if self._eval_type == 'rgb':
            dataset = dataset.map (
                lambda r_p,  l: tf.py_func (self._py_func_rgb_test_vp, [r_p, l], [ tf.float32, tf.float32]),
                num_parallel_calls=os.cpu_count())
        elif self._eval_type == 'flow':
            dataset = dataset.map (
                lambda f_p, l: tf.py_func (self._py_func_flow_test_vp, [f_p, l], [tf.float32, tf.float32]),
                num_parallel_calls=os.cpu_count())
        else:
            dataset = dataset.map (
                lambda r_p, f_p, l: tf.py_func (self._py_func_test_vp, [r_p, f_p, l], [tf.float32, tf.float32, tf.float32]),
                num_parallel_calls=os.cpu_count())
        print('dataset transformation successfully')
        dataset = dataset.batch(batch_size=self._batch_size)
        dataset = dataset.prefetch(buffer_size=self._prefetch_buffer_size)
        return dataset

    @staticmethod
    def _py_func_rgb_vp(rgb_path, label):

        rgb_path = rgb_path.decode ()

        # rgb_file = _rgb_vp (rgb_path)
        rgb_cap = cv2.VideoCapture (rgb_path)
        rgb_len = rgb_cap.get (cv2.CAP_PROP_FRAME_COUNT)

        image_list = []
        # while 1:
        try:
            index = np.random.randint (0, rgb_len-_frame_counts)
        except:
            index = 0

        rgb_cap.set (cv2.CAP_PROP_POS_FRAMES, index)
        while 1:
            flag, image = rgb_cap.read ()
            if image is not None:
                # image = np.float32 (image)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = cv2.resize (image, (340, 256))
                image_list.append(image)
            if len(image_list) == _frame_counts or flag is False:
                if len(image_list) == 0:
                    rgb_cap.set (cv2.CAP_PROP_POS_FRAMES, 0)    
                    continue
                break

        rgb_cap.release ()
        image = np.stack(image_list,axis=0)

        # some bug in rgb video,maybe you should use ffmpeg for image ,but read video is faster than read image due to IO spare
        assert 0 not in image.shape
        if _frame_counts != image.shape[0]:
            image = np.repeat(image,(_frame_counts//image.shape[0])+1,axis=0)[:_frame_counts,...]
        assert (_frame_counts,256,340,3) == image.shape

        if _crop == 'random':
            rgb_file = DataAugmentation.randomCrop(image,224,224)
            # print('random crop is used')
        else:
            rgb_file =  DataAugmentation.Multiscale_crop(image)
            # print('mutltisacle crop is used')
            rgb_file = np.float32([cv2.resize(x,(224,224)) for x in rgb_file])
        assert 0 not in rgb_file.shape
        rgb_file = DataAugmentation.horizontal_flip (rgb_file)

        rgb_file = np.float32(rgb_file)
        rgb_file = normalize (rgb_file)
        # rgb_file = (rgb_file / 255) * 2 - 1

        assert rgb_file is not None

        if label is not None:
            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return rgb_file, one_hot_label

        return rgb_file

    @staticmethod
    def _py_func_flow_vp(flow_path, label):

        flow_path = flow_path.decode ()

        flow_cap = cv2.VideoCapture (flow_path)
        flow_len = flow_cap.get (cv2.CAP_PROP_FRAME_COUNT)

        flag = True
        image_list = []
        try:
            index = np.random.randint (0, flow_len - _frame_counts)
        except:
            index = 0

        flow_cap.set (cv2.CAP_PROP_POS_FRAMES, index)
        for i in range(_frame_counts):
            flag, image = flow_cap.read ()
            if image is not None:
                image = np.float32 (image)
                image = cv2.resize (image, (340, 256))
                image_list.append (image[...,:2])
            if len (image_list) == _frame_counts or flag is False:
                break

        flow_cap.release ()
        image = np.stack (image_list, axis=0)
        assert 0 not in image.shape
        if _frame_counts != image.shape[0]:
            image = np.repeat(image,(_frame_counts//image.shape[0])+1,axis=0)[:_frame_counts,...]
        assert (_frame_counts,256,340,2) == image.shape
        # flow_file = DataAugmentation.randomCrop (image, 112, 112)
        # rgb_file = DataAugmentation.Multiscale_crop(image,is_flow=True)
        # flow_file = DataAugmentation.randomCrop(image,224,224)
        flow_file = DataAugmentation.borders25(image)
        flow_file = DataAugmentation.horizontal_flip (flow_file)
        flow_file = np.float32([cv2.resize(x,(224,224)) for x in flow_file])

        # image_list = []
        # for i in range(rgb_file.shape[0]):
        #     image_list.append(cv2.resize(flow_file[i],(224,224)))
        # flow_file = np.stack(image_list,axis=0)
        flow_file = np.float32(flow_file)
        flow_file = (flow_file / 255) * 2 - 1

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
        rgb_file = ucf_dataset._py_func_rgb_vp (rgb_path, label=None)
        flow_file = ucf_dataset._py_func_flow_vp (flow_path, label=None)

        one_hot_label = np.zeros (101, dtype=np.float32)
        one_hot_label[label] = 1

        return rgb_file, flow_file, one_hot_label

    
    @staticmethod
    def _py_func_rgb_test_vp(rgb_path, label):

        rgb_path = rgb_path.decode ()

        # rgb_file = _rgb_vp (rgb_path)
        rgb_cap = cv2.VideoCapture (rgb_path)
        rgb_len = rgb_cap.get (cv2.CAP_PROP_FRAME_COUNT)

        total_rgb_file = []
        try:
            index_list = np.arange(0,rgb_len-_frame_counts,(rgb_len-_frame_counts)//_batch_size)[:_batch_size]
            np.random.shuffle(index_list)
        except:
            if rgb_len > _frame_counts:
                index_list = np.repeat(np.arange(0,rgb_len-_frame_counts),repeats=(_batch_size//(rgb_len-_frame_counts)+1),axis=0)[:_batch_size]
            else:
                index_list = [0] * _frame_counts

        for index in index_list:
            image_list = []
            while 1:
                # index = np.random.randint (0, rgb_len - _frame_counts)
                rgb_cap.set (cv2.CAP_PROP_POS_FRAMES, index)
                for i in range(_frame_counts):
                    flag, image = rgb_cap.read ()
                    if image is not None:
                        # image = np.float32 (image)
                        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                        image = cv2.resize (image, (340, 256))
                        image_list.append(image)
                if len (image_list) == _frame_counts or flag is False:
                    break

            image = np.stack (image_list, axis=0)
            if image.shape[0] < _frame_counts:
                image = np.repeat(image,(_frame_counts//image.shape[0])+1,axis=0)[:_frame_counts,...]
            rgb_file = DataAugmentation.center_crop (image, 224, 224)
            rgb_file = np.float32 (rgb_file)
            rgb_file = normalize (rgb_file)
            rgb_file_flip = rgb_file[:,:,::-1,:]
            assert rgb_file is not None
            total_rgb_file.append(rgb_file)
            total_rgb_file.append(rgb_file_flip)

        rgb_cap.release ()
        if label is not None:
            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return total_rgb_file, one_hot_label

        return total_rgb_file

    @staticmethod
    def _py_func_flow_test_vp(flow_path, label):

        flow_path = flow_path.decode ()

        flow_cap = cv2.VideoCapture (flow_path)
        flow_len = flow_cap.get (cv2.CAP_PROP_FRAME_COUNT)

        total_flow_file = []
        try:
            index_list = np.arange(0,flow_len-_frame_counts,(flow_len-_frame_counts)//_batch_size)[:_batch_size]
        except:
            if flow_len > _frame_counts:
                index_list = np.repeat(np.arange(0,flow_len-_frame_counts),repeats=(_batch_size//(flow_len-_frame_counts)+1),axis=0)[:_batch_size]
            else:
                index_list = [0] * _frame_counts

        flag = True
        for index in index_list:
            image_list = []
                # index = np.random.randint (0, flow_len - 10)
            flow_cap.set (cv2.CAP_PROP_POS_FRAMES, index)
            for i in range(_frame_counts):
                flag, image = flow_cap.read ()
                if image is not None:
                    image = np.float32 (image)
                    image = cv2.resize (image, (340, 256))
                    image_list.append (image[...,:2])
                if len (image_list) == _frame_counts or flag is False:
                    break

            image = np.stack (image_list, axis=0)
            if image.shape[0] < _frame_counts:
                image = np.repeat(image,(_frame_counts//image.shape[0])+1,axis=0)[:_frame_counts,...]
            flow_file = DataAugmentation.center_crop (image, 224, 224)
            flow_file = (flow_file / 255) * 2 - 1
            # flow_file_flip = flow_file[:,:,::-1,:]
            total_flow_file.append(flow_file)
            # total_flow_file.append(flow_file_flip)

        if label is not None:
            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return total_flow_file, one_hot_label

        return total_flow_file

    @staticmethod
    def _py_func_test_vp(rgb_path, flow_path, label):
        # data augmentation parameter

        #############
        rgb_file = ucf_dataset._py_func_rgb_test_vp (rgb_path, label=None)
        flow_file = ucf_dataset._py_func_flow_test_vp (flow_path, label=None)

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
    import time
    import tensorflow as tf
    tf.enable_eager_execution()

    m_d = ucf_dataset (split_number=0, is_training_split=True,
                                    batch_size=100, epoch=1,
                                    frame_counts=3, eval_type='rgb',
                                    image_size=224,video_split=10,
                                    prefetch_buffer_size=20).dataset ()
    
    iter = m_d.make_one_shot_iterator()
    for i in range(50):
        t = time.time()
        g = iter.next()
        # r = np.squeeze(r)
        end_t = time.time() - t
        print(g[0].shape,end_t)
