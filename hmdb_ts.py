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
                 video_split = 25,
                 epoch=40,
                 new_length = 10,
                 prefetch_buffer_size=24,
                 eval_type='rgb',
                 test_crop = 'multi'):
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
        self._video_split = video_split
        self._epoch = epoch
        self._eval_type = eval_type
        self._test_crop = test_crop
        self._new_length = new_length
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
        global _test_crop
        global _new_length 
        _IMAGE_SIZE = self._IMAGE_SIZE
        _batch_size = self._video_split
        _frame_counts = self._FRAME_COUNTS
        _test_crop = self._test_crop
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


    def test_dataset(self):
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
    def pick_index(length,counts,is_temporal_segment=True,is_flow=False):
        if is_flow:
            g = np.array_split(np.arange(length),counts)
            index_list = []
            for i in g:
                index_list.append(np.random.choice(i))
            return index_list
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
    def _py_func_rgb_vp(rgb_path, label , _index_list = None):

        rgb_path = rgb_path.decode ()

        rgb_cap = cv2.VideoCapture (rgb_path)
        rgb_len = rgb_cap.get (cv2.CAP_PROP_FRAME_COUNT)

        image_list = []
        
        if _index_list is None:
            index_list = hmdb_dataset.pick_index(rgb_len,_frame_counts,is_flow=True)
        else:
            index_list = _index_list

        for index in index_list:
            rgb_cap.set (cv2.CAP_PROP_POS_FRAMES, index)
            flag, image = rgb_cap.read ()
            if image is not None:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = np.float32(image)
                image = cv2.resize (image, (340, 256))
                image = DataAugmentation.Multiscale_crop(image)
                image = cv2.resize(image,(224,224))
                image = DataAugmentation.horizontal_flip (image)
                image = normalize(image)
                image_list.append(image)

        if len(image_list) != _frame_counts:
            image_list = np.concatenate([image_list] * _frame_counts,axis=0)[:_frame_counts]   

        rgb_cap.release ()
        np.random.shuffle(image_list)
        rgb_file = np.stack(image_list,axis=0)

        if label is not None:
            one_hot_label = np.zeros (51, dtype=np.float32)
            one_hot_label[label] = 1

            return rgb_file, one_hot_label

        return rgb_file

    @staticmethod
    def _py_func_flow_vp(flow_path, label , _index_list = None):

        f_upath , f_vpath = flow_path
        flow_u_path = f_upath.decode ()
        flow_v_path = f_vpath.decode ()
        flow_file = os.listdir (flow_u_path)
        flow_file = sorted(flow_file)
        flow_len = len(flow_file)

        if _index_list is None:
            index_list = hmdb_dataset.pick_index(flow_len-_new_length,_frame_counts,is_flow=True)
        else:
            index_list = _index_list

        flow_img = []
        for j in index_list:
            img_list = []
            for i in range(j,j+_new_length):
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
            img = DataAugmentation.horizontal_flip(img)
            img = cv2.resize(img,(224,224))
            img = np.float32(img)
            img = (img / 255 - 0.5) / 0.226
            flow_img.append(img)

        np.random.shuffle(flow_img)
        if label is not None:

            one_hot_label = np.zeros (51, dtype=np.float32)
            one_hot_label[label] = 1

            return flow_img, one_hot_label

        return flow_img

    @staticmethod
    def _py_func_vp(rgb_path, flow_path, label):
        
        # f_upath , f_vpath = flow_path
        # flow_u_path = f_upath.decode ()
        # flow_file = os.listdir (flow_u_path)
        # flow_len = len(flow_file)
        # index_list = hmdb_dataset.pick_index(flow_len-_new_length,_frame_counts,is_flow=True)
        #############
        rgb_file = hmdb_dataset._py_func_rgb_vp (rgb_path, label=None)
        flow_file = hmdb_dataset._py_func_flow_vp (flow_path, label=None)

        one_hot_label = np.zeros (51, dtype=np.float32)
        one_hot_label[label] = 1

        return rgb_file, flow_file, one_hot_label

    
    @staticmethod
    def _py_func_rgb_test_vp(rgb_path, label):

        rgb_path = rgb_path.decode ()

        rgb_cap = cv2.VideoCapture (rgb_path)
        rgb_len = rgb_cap.get (cv2.CAP_PROP_FRAME_COUNT)

        total_rgb_file = []
        try:
            index_list = np.arange(0,rgb_len,rgb_len//_frame_counts)[:_frame_counts]
        except:
            index_list = np.arange(0,rgb_len)[:_frame_counts]

        image_list = []
        for index in index_list:
            rgb_cap.set (cv2.CAP_PROP_POS_FRAMES, index)
            _ , image = rgb_cap.read ()
            if image is not None:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = cv2.resize (image, (340, 256))
                image_list.append(image)

        if len(image_list) != _frame_counts:
            image_list = np.concatenate([image_list] * _frame_counts,axis=0)[:_frame_counts]

        image = np.stack (image_list, axis=0)

        img = image
        if _test_crop == 'center':
            rgb_file = DataAugmentation.center_crop (img, 224, 224)
            rgb_file = np.float32 (rgb_file)
            rgb_file = normalize (rgb_file)
            rgb_file_flip = rgb_file[:,:,::-1,:]
            total_rgb_file.append(rgb_file)
            total_rgb_file.append(rgb_file_flip)
        else:
            for k in range(5):
                image = DataAugmentation.random_Crop(img,1,k)
                image = np.float32(image)
                image = normalize (image)
                image_flip = image[:,:,::-1,:]
                total_rgb_file.append(image)
                total_rgb_file.append(image_flip)

        rgb_cap.release ()
        if label is not None:
            one_hot_label = np.zeros (51, dtype=np.float32)
            one_hot_label[label] = 1

            return total_rgb_file, one_hot_label

        return total_rgb_file

    @staticmethod
    def _py_func_flow_test_vp(flow_path, label):

        f_upath , f_vpath = flow_path
        flow_u_path = f_upath.decode ()
        flow_v_path = f_vpath.decode ()
        flow_file = os.listdir (flow_u_path)
        flow_file = sorted(flow_file)

        flow_len = len(flow_file)
        try:
            index_list = np.arange(0,flow_len - _new_length,(flow_len - _new_length)//_frame_counts)[:_frame_counts]
        except:
            index_list = np.arange(0,flow_len - _new_length)[:_frame_counts]


        total_img_list = []
        video_list = []
        for index in index_list:
            img_list = []
            for i in range(index,index + _new_length):
                img_u_path = os.path.join(flow_u_path,flow_file[i])
                img_v_path = os.path.join(flow_v_path,flow_file[i])
                
                img_u = cv2.imread(img_u_path,0)
                img_v = cv2.imread(img_v_path,0)
                
                if img_u is not None and img_v is not None:
                    img = np.stack([img_u,img_v],axis=-1)
                    img = cv2.resize(img,(340,256))
                    img_list.append(img)

            img = np.concatenate(img_list,axis=-1)
            video_list.append(img)
        
        if len(video_list) != _frame_counts:
            video_list = np.concatenate([video_list] * _frame_counts,axis=0)[:_frame_counts]

        img = np.stack(video_list,axis=0)

        if _test_crop == 'center':
            image = DataAugmentation.center_crop(img,224,224)
            image = np.float32(image)
            image = (image / 255 - 0.5) / 0.226
            image_flip = image[:,:,::-1,:]
            total_img_list.append(image)
            total_img_list.append(image_flip)
        else:
            for k in range(5):
                image = DataAugmentation.random_Crop(img,1,k)
                image = np.float32(image)
                image = (image / 255 - 0.5) / 0.226
                image_flip = image[:,:,::-1,:]
                total_img_list.append(image)
                total_img_list.append(image_flip)
        
        if label is not None:

            one_hot_label = np.zeros (51, dtype=np.float32)
            one_hot_label[label] = 1

            return total_img_list, one_hot_label

        return total_img_list

    @staticmethod
    def _py_func_test_vp(rgb_path, flow_path, label):

        rgb_file = hmdb_dataset._py_func_rgb_test_vp (rgb_path, label=None)
        flow_file = hmdb_dataset._py_func_flow_test_vp (flow_path, label=None)

        one_hot_label = np.zeros (51, dtype=np.float32)
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
    m = hmdb_dataset(batch_size=1,eval_type='rgb',video_split=2,prefetch_buffer_size=1,frame_counts=25,test_crop='multi').dataset()
    # iter = tf.data.Iterator.from_structure (m.output_types, m.output_shapes)
    iter = m.make_one_shot_iterator()
    g = iter.get_next()
    print(g[0].shape,g[1].shape)