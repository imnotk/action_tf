from tensorflow.contrib.slim.nets import vgg,resnet_v1,resnet_v2,inception
import tensorflow as tf
from tensorflow.contrib.framework import load_variable,list_variables
import numpy as np
import scipy.io as sio
import os

def rebuild_rgb_resnet_ckpt(checkpoint_dir,save_path):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            raw_var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            #             if var_name.find('logits') > -1 or var_name.find('Logits')  > -1:
            #                 continue
            print(var_name, raw_var.shape)
            var = tf.Variable(raw_var, name=  'RGB/'+var_name)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess,save_path)


def rebuild_Flow_resnet_ckpt(checkpoint_dir,save_path):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            raw_var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            #             if var_name.find('logits') > -1 or var_name.find('Logits')  > -1:
            #                 continue
            if var_name == 'resnet_v1_152/conv1/weights' or var_name == 'resnet_v1_50/conv1/weights':
                b = np.mean(raw_var,axis=2,keepdims=True)
                raw_var = np.repeat(b,10,axis=2)
            print(var_name, raw_var.shape)
            var = tf.Variable(raw_var, name=  'Flow/'+var_name)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess,save_path)


def rebuild_rgb_vgg_ckpt(checkpoint_dir,save_path):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            raw_var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            #             if var_name.find('logits') > -1 or var_name.find('Logits')  > -1:
            #                 continue
            print(var_name, raw_var.shape)
            var = tf.Variable(raw_var, name=  'RGB/'+var_name)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess,save_path)


def rebuild_Flow_vgg_ckpt(checkpoint_dir,save_path):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            raw_var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            #             if var_name.find('logits') > -1 or var_name.find('Logits')  > -1:
            #                 continue
            if var_name == 'vgg_16/conv1/conv1_1/weights':
                b = np.mean(raw_var,axis=2,keepdims=True)
                raw_var = np.repeat(b,20,axis=2)
            print(var_name, raw_var.shape)
            var = tf.Variable(raw_var, name=  'Flow/'+var_name)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess,save_path)

name_map = {
    'weights':'conv_2d/w',
    'BatchNorm':'batch_norm',
    'biases':'conv_2d/b'
}

def rebuild_rgb_inceptionv1_ckpt(checkpoint_dir,save_path):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            raw_var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            #             if var_name.find('logits') > -1 or var_name.find('Logits')  > -1:
            #                 continue
            print(var_name, raw_var.shape)
            for k ,v in name_map.items():
                var_name=var_name.replace(k,v)
            var = tf.Variable(raw_var, name=  'RGB/'+var_name)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess,save_path)

def rebuild_Flow_inceptionv1_ckpt(checkpoint_dir,save_path):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            raw_var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            #             if var_name.find('logits') > -1 or var_name.find('Logits')  > -1:
            #                 continue
            print(var_name, raw_var.shape)
            if var_name == 'InceptionV1/Conv2d_1a_7x7/weights':
                b = np.mean(raw_var,axis=2,keepdims=True)
                raw_var = np.repeat(b,20,axis=2)
            for k ,v in name_map.items():
                var_name = var_name.replace(k,v)
            var = tf.Variable(raw_var, name=  'Flow/'+var_name)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess,save_path)

def rebuild_Flow_snt_resnetV1_ckpt(checkpoint_dir,save_path):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            raw_var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            #             if var_name.find('logits') > -1 or var_name.find('Logits')  > -1:
            #                 continue
            print(var_name, raw_var.shape)
            if var_name == 'resnet_v1_152/conv1/weights' or var_name == 'resnet_v1_50/conv1/weights' :
                b = np.mean(raw_var,axis=2,keepdims=True)
                raw_var = np.repeat(b,2,axis=2)
            for k ,v in name_map.items():
                var_name = var_name.replace(k,v)
            var = tf.Variable(raw_var, name=  'Flow/'+var_name)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess,save_path)

def show_ckpt_var(ckpt_dir):
    for i in tf.contrib.framework.list_variables(ckpt_dir):
        print(i)


def restore_ckpt_to_net(ckpt_dir,_net):
    sample = tf.placeholder(tf.float32,[None,224,224,3])
    _ ,e = _net(sample,num_classes = 1000)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,ckpt_dir)



if __name__ == '__main__':
    rebuild_Flow_snt_resnetV1_ckpt("/mnt/zhujian/ckpt/resnet_v1_50_2016_08_28/resnet_v1_50.ckpt",'/mnt/zhujian/ckpt/2frame_flow_snt_resnetV1_50/model.ckpt')