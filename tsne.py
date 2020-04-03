from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from sklearn.manifold import TSNE
import tensorflow as tf
# from model import TS_resnet , ts_inception_v1
# from model import ts_fusion_net
# from model import ts_non_local_fusion_net
import os
import time
import numpy as np
import ucf_ts,hmdb_ts
import pickle
import matplotlib.pyplot as plt


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gpu_options = tf.GPUOptions (per_process_gpu_memory_fraction=0.98, allow_growth=True)
file_save_path = '/mnt/zhujian/action_recognition'
log_dir = 'log_dir_with_two_stream_m2d/'
tf.app.flags.DEFINE_string ('video_dir', './UCF-101/', 'the orginal and optical video directory')
tf.app.flags.DEFINE_integer ('image_size', 224, 'the uniform input video size')
tf.app.flags.DEFINE_integer ('num_classes', 101, 'the classes number of the video dataset')
tf.app.flags.DEFINE_integer ('frame_counts', 10, 'the input video frame counts')
tf.app.flags.DEFINE_integer ('batch_size', 1, 'the inputed batch size ')
tf.app.flags.DEFINE_integer ('gpu_nums', 2, 'the inputed batch size ')
tf.app.flags.DEFINE_string ('dataset', 'UCF101', 'the training dataset')
tf.app.flags.DEFINE_string ('eval_type', 'fusion', 'fusion or joint')
tf.app.flags.DEFINE_string ('fusion_type','non_local','fhn or fpn or non local')
tf.app.flags.DEFINE_integer ('split', 0, 'split number ')
tf.app.flags.DEFINE_string ('model_type', 'resnet50', 'use which model vgg,inception,resnet and so on')
tf.app.flags.DEFINE_string ('fusion_mode', 'add', 'use add or concat to fusion rgb and flow feature')
tf.app.flags.DEFINE_string ('test_crop', 'center', 'use center or multiscale crop')
tf.app.flags.DEFINE_integer('video_split',25,'video split')
tf.app.flags.DEFINE_boolean ('use_space', True, 'use space time fusion or not')


Flags = tf.app.flags.FLAGS

split = Flags.split
eval_type = Flags.eval_type
gpu_nums = Flags.gpu_nums
model_type = Flags.model_type
fusion_type = Flags.fusion_type
test_crop = Flags.test_crop
fusion_mode = Flags.fusion_mode
video_split = Flags.video_split
test_crop = Flags.test_crop
space = Flags.use_space
batch_size = Flags.batch_size
_FRAME_COUNTS = Flags.frame_counts
_IMAGE_SIZE = Flags.image_size
_NUM_CLASSES = Flags.num_classes
dataset_name = Flags.dataset


if model_type == 'resnet50':
    base_net = 'TS_resnet50'
elif model_type == 'resnet101':
    base_net = 'ts_resnet101'
elif model_type == 'inception_v1':
    base_net = 'ts_inception_v1'


train_log_path = os.path.join (log_dir, dataset_name, str (split), base_net, eval_type,fusion_type,fusion_mode)


def plot_embedding(data, label, title):
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(label[i]),
    #              color=plt.cm.Set1(label[i] / 10.),
    #              fontdict={'weight': 'bold', 'size': 9})
    label = np.argmax(label,axis=1)
    print(label.shape)
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.Spectral)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


tsne = TSNE(n_components=2,init='pca')
with open(os.path.join(train_log_path,'video_tsne_%s.pickle' % test_crop),'rb') as f:

    rgb = open(os.path.join(train_log_path,'rgb_video_tsne_%s.pickle' % test_crop),'rb')
    flow = open(os.path.join(train_log_path,'flow_video_tsne_%s.pickle' % test_crop),'rb')
    fusion_a = pickle.load(f)
    rgb_a = pickle.load(rgb)
    flow_a = pickle.load(flow)
    
    b = fusion_a[:300]
    rgb_b = rgb_a[:300]
    flow_b = flow_a[:300]

    label = []
    feature = []
    rgb_feature = []
    flow_feature = []
    for i in b:
        label.append(i[0])
        feature.append(np.squeeze(i[1]))
    
    for i in rgb_b:
        rgb_feature.append(np.squeeze(i[1]))
    for i in flow_b:
        flow_feature.append(np.squeeze(i[1]))
    
    label = np.array(label)
    feature = np.array(feature)
    t0 = time.time()
    y = tsne.fit_transform(feature)
    t1 = time.time()
    fig = plot_embedding(y, label, title='t-sne embedding of the ucf101')    
    print('tsne: %.2g sec' % (t1 - t0))
    plt.savefig(os.path.join(train_log_path,'fusion.jpg'))
    plt.show(fig)
    

    y = tsne.fit_transform(rgb_feature)
    fig = plot_embedding(y, label, title='rgb t-sne embedding of the ucf101')    
    plt.savefig(os.path.join(train_log_path,'rgb.jpg'))
    plt.show(fig)

    y = tsne.fit_transform(flow_feature)
    fig = plot_embedding(y, label, title='flow t-sne embedding of the ucf101')
    plt.savefig(os.path.join(train_log_path,'flow.jpg'))

    plt.show(fig)
