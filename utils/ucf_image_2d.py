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
                 video_path_include_label = '/mnt/zhujian/jpegs_256/',
                 flow_u_path = 'ucf101_tvl1_flow/tvl1_flow/u',
                 flow_v_path = 'ucf101_tvl1_flow/tvl1_flow/v',
                 split_number=0,
                 is_training_split=True,
                 frame_counts=10,
                 image_size=224,
                 batch_size = 24,
                 epoch=40,
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
        _IMAGE_SIZE = self._IMAGE_SIZE
        _batch_size = self._batch_size
        _frame_counts = self._FRAME_COUNTS

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
                path = trainlist[i][0][:-4]
                label_list.append( int(trainlist[i][1]) - 1)
                all_path.append (path)
            train_split[index] = (all_path, label_list)
            index += 1
        index = 0
        for testlist in [testlist1, testlist2, testlist3]:
            label_list = []
            all_path = []
            for i in range (testlist.shape[0]):
                path = testlist[i][:-4]
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
        flow_u_path = [os.path.join (self._flow_u_path, path) for path in self._path]
        flow_v_path = [os.path.join (self._flow_v_path, path) for path in self._path]
        label = list(self._label)
        rgb_dataset = tf.data.Dataset.from_tensor_slices ((rgb_path))
        flow_dataset = tf.data.Dataset.from_tensor_slices ((flow_u_path,flow_v_path))
        label_dataset = tf.data.Dataset.from_tensor_slices ((label))
        print('dataset create successfully')
        if self._eval_type == 'rgb':
            dataset = tf.data.Dataset.zip((rgb_dataset,label_dataset))
        elif self._eval_type == 'flow':
            dataset = tf.data.Dataset.zip((flow_dataset,label_dataset))
        else:
            dataset = tf.data.Dataset.zip((rgb_dataset,flow_dataset,label_dataset))
        print('image dataset create successfully')
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
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=self._batch_size))
        dataset = dataset.prefetch(buffer_size=self._prefetch_buffer_size)
        return dataset

    def test_dataset(self):
        rgb_path = [os.path.join(self._videl_path_include_label,path) for path in self._path]
        flow_u_path = [os.path.join (self._flow_u_path, path) for path in self._path]
        flow_v_path = [os.path.join (self._flow_v_path, path) for path in self._path]

        label = list (self._label)
        rgb_dataset = tf.data.Dataset.from_tensor_slices ((rgb_path))
        flow_dataset = tf.data.Dataset.from_tensor_slices ((flow_u_path,flow_v_path))
        label_dataset = tf.data.Dataset.from_tensor_slices ((label))
        if self._eval_type == 'rgb':
            dataset = tf.data.Dataset.zip ((rgb_dataset, label_dataset))
        elif self._eval_type == 'flow':
            dataset = tf.data.Dataset.zip ((flow_dataset, label_dataset))
        else:
            dataset = tf.data.Dataset.zip ((rgb_dataset, flow_dataset, label_dataset))
        print ('dataset create successfully')
        dataset = dataset.repeat (self._epoch)
        if self._eval_type == 'rgb':
            dataset = dataset.map (
                lambda r_p, l: tf.py_func (self._py_func_rgb_test_vp, [r_p, l], [tf.float32, tf.float32]),
                num_parallel_calls=os.cpu_count())
        elif self._eval_type == 'flow':
            dataset = dataset.map (
                lambda f_p, l: tf.py_func (self._py_func_flow_test_vp, [f_p, l], [tf.float32, tf.float32]),
                num_parallel_calls=os.cpu_count())
        else:
            dataset = dataset.map (
                lambda r_p, f_p, l: tf.py_func (self._py_func_test_vp, [r_p, f_p, l],
                                                [tf.float32, tf.float32, tf.float32]),
                num_parallel_calls=os.cpu_count())
        print ('dataset transformation successfully')
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=self._batch_size))
        dataset = dataset.prefetch (buffer_size=self._prefetch_buffer_size)
        return dataset

    @staticmethod
    def _py_func_rgb_vp(rgb_path, label):

        rgb_path = rgb_path.decode ()

        rgb_file = os.listdir (rgb_path)
        rgb_file = sorted(rgb_file)
        try:
            random_file = random.choice(rgb_file)
            rgb_file_path = os.path.join(rgb_path,random_file)
            img = cv2.imread(rgb_file_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        except:
            raise ValueError('rgb_file_path',rgb_file_path,'cannot be read')

        img = cv2.resize(img,(340,256))
        img = DataAugmentation.Multiscale_crop(img)
        img = DataAugmentation.horizontal_flip(img)
        img = cv2.resize(img,(224,224))
        img = np.float32(img)
        img = normalize(img)

        assert img.shape == (224,224,3)

        if label is not None:
            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return img, one_hot_label

        return img

    @staticmethod
    def _py_func_flow_vp(f_path, label):

        f_upath , f_vpath = f_path
        flow_u_path = f_upath.decode ()
        flow_v_path = f_vpath.decode ()
        flow_file = os.listdir (flow_u_path)
        flow_file = sorted(flow_file)
        index = np.random.randint(0,len(flow_file)-10)
        img_list = []
        for i in range(index,index+10):
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
        img = DataAugmentation.Multiscale_crop(img,is_flow=True)
        # img = DataAugmentation.borders25(img)
        img = DataAugmentation.horizontal_flip(img)
        img = cv2.resize(img,(224,224))
        img = np.float32(img)
        img = (img / 255 - 0.5) / 0.226

        if label is not None:

            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return img, one_hot_label

        return img

    @staticmethod
    def _py_func_vp(rgb_path, flow_path, label):

        rgb_path = rgb_path.decode ()
        flow_path = flow_path.decode ()
        rgb_file = ucf_dataset._py_func_rgb_vp (rgb_path,None)
        flow_file = ucf_dataset._py_func_flow_vp (flow_path,None)


        one_hot_label = np.zeros (101, dtype=np.float32)
        one_hot_label[label] = 1

        return rgb_file, flow_file, one_hot_label

    @staticmethod
    def _py_func_rgb_test_vp(rgb_path, label):

        rgb_path = rgb_path.decode ()

        rgb_file = os.listdir (rgb_path)
        rgb_file = sorted(rgb_file)

        _batch_size = 25

        if len(rgb_file) < _batch_size:
            index_list = np.arange(0,len(rgb_file))
            index_list = np.concatenate([index_list]*(_batch_size//len(rgb_file) + 1),axis=0)[:_batch_size]
        else:
            index_list = np.arange(0,len(rgb_file),len(rgb_file)//_batch_size)[:_batch_size]

        img_list = []
        for i in index_list:
            rgb_file_path = os.path.join(rgb_path,rgb_file[i])

            img = cv2.imread(rgb_file_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(340,256))
            for j in range(5):
                image = DataAugmentation.random_Crop(img,1,j)
                image = cv2.resize(image,(224,224))
                image = np.float32(image)
                image = normalize(image)
                image_flip = np.fliplr(image)
                img_list.append(image)
                img_list.append(image_flip)


        if label is not None:
            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return img_list, one_hot_label

        return img_list

    @staticmethod
    def _py_func_flow_test_vp(flow_path, label):

        f_upath , f_vpath = flow_path
        flow_u_path = f_upath.decode ()
        flow_v_path = f_vpath.decode ()
        flow_file = os.listdir (flow_u_path)
        flow_file = sorted(flow_file)

        _batch_size = 25
        if len(flow_file) - 10 < _batch_size:
            index_list = np.arange(0,len(flow_file)-10)
            index_list = np.concatenate([index_list]*(_batch_size//(len(flow_file)-10) + 1),axis=0)[:_batch_size]
        else:
            index_list = np.arange(0,len(flow_file)-10,(len(flow_file)-10)//_batch_size)[:_batch_size]


        total_img_list = []
        for index in index_list:
            img_list = []
            for i in range(index,index+10):
                img_u_path = os.path.join(flow_u_path,flow_file[i])
                img_v_path = os.path.join(flow_v_path,flow_file[i])
                
                img_u = cv2.imread(img_u_path,0)
                img_v = cv2.imread(img_v_path,0)
                
                if img_u is not None and img_v is not None:
                    img = np.stack([img_u,img_v],axis=-1)
                    img_list.append(img)

            img = np.concatenate(img_list,axis=-1)
            if img.shape[-1] != 20:
                img = np.concatenate([img]*(20//img.shape[-1]+1),axis=-1)

            img = cv2.resize(img,(340,256))
            image = DataAugmentation.center_crop(img,224,224)
            image = cv2.resize(image,(224,224))
            image = np.float32(image)
            image = (image / 255 - 0.5) / 0.226
            image_flip = np.fliplr(image)
            total_img_list.append(image)
            total_img_list.append(image_flip)
            # for j in range(5):
            #     image = DataAugmentation.random_Crop(img,1,j)
            #     # img = DataAugmentation.horizontal_flip(img)
            #     image = cv2.resize(image,(224,224))
            #     image = np.float32(image)
            #     image = (image / 255 - 0.5) / 0.226
            #     image_flip = np.fliplr(image)
            #     total_img_list.append(image)
            #     total_img_list.append(image_flip)
           
            


        if label is not None:

            one_hot_label = np.zeros (101, dtype=np.float32)
            one_hot_label[label] = 1

            return total_img_list, one_hot_label

        return total_img_list

    @staticmethod
    def _py_func_test_vp(rgb_path, flow_path, label):
        # data augmentation parameter

        #############
        rgb_path = rgb_path.decode ()
        flow_path = flow_path.decode ()
        rgb_file = ucf_dataset._py_func_rgb_test_vp (rgb_path,None)
        flow_file = ucf_dataset._py_func_flow_test_vp (flow_path,None)


        one_hot_label = np.zeros (101, dtype=np.float32)
        one_hot_label[label] = 1

        return rgb_file, flow_file, one_hot_label




def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = img / 255
    for i in range (3):
        img[..., i] = (img[..., i] - mean[i]) / std[i]
    return img


if __name__ == '__main__':
    from tensorflow.contrib.slim.nets import resnet_v1
    import time
    import tensorflow as tf
    tf.enable_eager_execution()

    m_d = ucf_dataset (split_number=0, is_training_split=True,
                                    batch_size=1, epoch=10,
                                    frame_counts=25, eval_type='flow',
                                    image_size=224,
                                    prefetch_buffer_size=20).dataset ()
    
    iter = m_d.make_one_shot_iterator()
    for i in range(10000):
        t = time.time()
        g = iter.next()
        # r = np.squeeze(r)
        end_t = time.time() - t
        print(g[0].shape,end_t)