import pickle
import numpy as np 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--rgb','-r',type=float,default=1.0)
parser.add_argument('--flow','-f',type=float,default=1.0)
parser.add_argument('--fusion','-fn',type=float,default=1.0)
parser.add_argument('--center','-c',type=int,default=1) 
parser.add_argument('--non_local','-nl',type=int,default=1)

args = parser.parse_args()
r_ratio = args.rgb
f_ratio = args.flow
fn_ratio = args.fusion
use_center = args.center
nl = args.non_local

if use_center == 0:
    f = open("/home/zhujian/video_analysi/action_recognition/log_dir/UCF101/0/se_resnet50/rgb/video_predict_multi.pickle",'rb')
    g = open("/home/zhujian/video_analysi/action_recognition/log_dir/UCF101/0/se_resnet50/flow/video_predict_multi.pickle",'rb')
    if nl == 0:
        fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/UCF101/0/se_resnet50/fusion/fhn/add/video_predict_multi.pickle",'rb')
        print('use fhn fusion')
    else:
        fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/UCF101/0/se_resnet50/fusion/non_local/add/video_predict_multi.pickle",'rb')
        print('use non local  fusion')
    print('use ten crop')
else:
    f = open("/home/zhujian/video_analysi/action_recognition/log_dir/UCF101/0/se_resnet50/rgb/video_predict_center.pickle",'rb')
    g = open("/home/zhujian/video_analysi/action_recognition/log_dir/UCF101/0/se_resnet50/flow/video_predict_center.pickle",'rb')
    if nl == 0:
        fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/UCF101/0/se_resnet50/fusion/fhn/add/video_predict_center.pickle",'rb')
    else:
        fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/UCF101/0/se_resnet50/fusion/non_local/add/video_predict_center.pickle",'rb')
    print('use center crop')

a = pickle.load(f)
b = pickle.load(g)
c = pickle.load(fusion)

video_predict = []
video_predict = []
for n , v in enumerate(a):
    label = a[n][0]
    rgb_logit = a[n][1]
    flow_logit = b[n][1]
    fusion_logit = c[n][1]
    total_logit = r_ratio * rgb_logit +  f_ratio * flow_logit + fn_ratio * fusion_logit
    # total_logit = r_ratio * rgb_logit +  f_ratio * flow_logit
    
    if np.argmax(total_logit) == np.argmax(label):
        video_predict.append(1)
    else:
        video_predict.append(0)

print(np.mean(video_predict))
# print(a)
# print(b)
