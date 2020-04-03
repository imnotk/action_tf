import pickle
import numpy as np 
import argparse
# import tensorflow as tf
# tf.enable_eager_execution()

def softmax(input):
    input = input - np.max(input)
    output = np.exp(input)
    output = output / np.sum(output)
    return output

parser = argparse.ArgumentParser()
parser.add_argument('--rgb',type=float,default=1.0)
parser.add_argument('--flow',type=float,default=1.0)
parser.add_argument('--center',type=int,default=1) 

args = parser.parse_args()
r_ratio = args.rgb
f_ratio = args.flow

f = open("/home/zhujian/video_analysi/zhujian/action_recognition/log_dir/UCF101/0/i3d/rgb/video_predict.pickle",'rb')
g = open("/home/zhujian/video_analysi/zhujian/action_recognition/log_dir/UCF101/0/i3d/flow/video_predict.pickle",'rb')


a = pickle.load(f)
b = pickle.load(g)

video_predict = []
video_softmax_predict = []
for n , v in enumerate(a):
    label = a[n][0]
    rgb_logit = a[n][1]
    flow_logit = b[n][1]
    total_logit = r_ratio * rgb_logit +  f_ratio * flow_logit
    total_softmax = r_ratio * softmax(rgb_logit) + f_ratio * softmax(flow_logit)
    if np.argmax(total_logit) == np.argmax(label):
        video_predict.append(1)
    else:
        video_predict.append(0)

    if np.argmax(total_softmax) == np.argmax(label):
        video_softmax_predict.append(1)
    else:
        video_softmax_predict.append(0)

print(np.mean(video_predict))
print('softmax score is',np.mean(video_softmax_predict))
# print(a)
# print(b)
