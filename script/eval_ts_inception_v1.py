import pickle
import numpy as np 
import argparse
# import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--rgb','-r',type=float,default=1.0)
parser.add_argument('--flow','-f',type=float,default=1.0)
parser.add_argument('--fusion','-fn',type=float,default=1.0)
parser.add_argument('--non_local','-nl',type=int,default=1)
parser.add_argument('--center','-c',type=int,default=1) 
parser.add_argument('--split','-s',type=int,default=0) 
parser.add_argument('--dataset','-d',type=str,default='UCF101') 


args = parser.parse_args()
r_ratio = args.rgb
f_ratio = args.flow
use_center = args.center
split = args.split
fusion_ratio = args.fusion
nl = args.non_local
dataset = args.dataset

if use_center == 0 :
    f = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/ts_inception_v1/rgb/video_predict_multi.pickle",'rb')
    g = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/ts_inception_v1/flow/video_predict_multi.pickle",'rb')
    print('use ten crop')
    if fusion_ratio != 0:
        if nl == 0:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/ts_inception_v1/fusion/fhn/"+'add'+"/video_predict_multi.pickle",'rb')
            print('use local fusion')
        else:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/ts_inception_v1/fusion/non_local/"+'add'+"/video_predict_multi.pickle",'rb')
            print('use non-local fusion')
else:
    print('use center crop')
    f = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/ts_inception_v1/rgb/video_predict_center.pickle",'rb')
    g = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/ts_inception_v1/flow/video_predict_center.pickle",'rb')
    if fusion_ratio != 0:
        if nl == 0:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/ts_inception_v1/fusion/fhn/"+'add'+"/video_predict_center.pickle",'rb')
            print('use local fusion')
        else:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/ts_inception_v1/fusion/non_local/"+'add'+"/video_predict_center.pickle",'rb')
            print('use non-local fusion')

a = pickle.load(f)
b = pickle.load(g)
if fusion_ratio != 0:
    c = pickle.load(fusion)

video_predict = []
video_softmax_predict = []
for n , v in enumerate(a):
    label = a[n][0]
    rgb_logit = a[n][1]
    flow_logit = b[n][1]
    if fusion_ratio != 0:
        fusion_logit = c[n][1]
        total_logit = r_ratio * rgb_logit +  f_ratio * flow_logit + fusion_ratio * fusion_logit
    else:
        total_logit = r_ratio * rgb_logit +  f_ratio * flow_logit
        
    if np.argmax(total_logit) == np.argmax(label):
        video_predict.append(1)
    else:
        video_predict.append(0)


print(np.mean(video_predict))
# print(a)
# print(b)
