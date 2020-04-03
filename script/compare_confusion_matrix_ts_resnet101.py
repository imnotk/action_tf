import pickle
import numpy as np 
import argparse
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--rgb','-r',type=float,default=1.0)
parser.add_argument('--flow','-f',type=float,default=1.0)
parser.add_argument('--fusion','-fn',type=float,default=1.0)
parser.add_argument('--non_local','-nl',type=int,default=1)
parser.add_argument('--center','-c',type=int,default=1) 
parser.add_argument('--split','-s',type=int,default=0) 
parser.add_argument('--fusion_mode','-fm',type=str,default='add') 
parser.add_argument('--dataset','-d',type=str,default='UCF101') 


args = parser.parse_args()
r_ratio = args.rgb
f_ratio = args.flow
use_center = args.center
split = args.split
fusion_ratio = args.fusion
nl = args.non_local
fm = args.fusion_mode
dataset = args.dataset

if use_center == 0 :
    f = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/ts_resnet101/rgb/video_predict_multi.pickle",'rb')
    g = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/ts_resnet101/flow/video_predict_multi.pickle",'rb')
    print('use ten crop')
    if fusion_ratio != 0:
        if nl == 0:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/ts_resnet101/fusion/fhn/"+fm+"/video_predict_multi.pickle",'rb')
            print('use local fusion')
        else:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/ts_resnet101/fusion/non_local/"+fm+"/video_predict_multi.pickle",'rb')
            print('use non-local fusion')
else:
    print('use center crop')
    f = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/ts_resnet101/rgb/video_predict_center.pickle",'rb')
    g = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/ts_resnet101/flow/video_predict_center.pickle",'rb')
    if fusion_ratio != 0:
        if nl == 0:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/ts_resnet101/fusion/fhn/"+fm+"/video_predict_center.pickle",'rb')
            print('use local fusion')
        else:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/ts_resnet101/fusion/non_local/"+fm+"/video_predict_center.pickle",'rb')
            print('use non-local fusion')

a = pickle.load(f)
b = pickle.load(g)
if fusion_ratio != 0:
    c = pickle.load(fusion)

video_predict = []
video_fusion_predict = []
video_softmax_predict = []
video_label = []
for n , v in enumerate(a):
    label = a[n][0]
    rgb_logit = a[n][1]
    flow_logit = b[n][1]
    fusion_logit = c[n][1]
    total_fusion_logit = r_ratio * rgb_logit +  f_ratio * flow_logit + fusion_ratio * fusion_logit
    total_logit = r_ratio * rgb_logit +  f_ratio * flow_logit
    # if np.argmax(total_logit) == np.argmax(label):
    #     video_predict.append(1)
    # else:
    #     video_predict.append(0)
    video_fusion_predict.append(np.argmax(total_fusion_logit))
    video_predict.append(np.argmax(total_logit))
    video_label.append(np.argmax(label))




classind = np.genfromtxt ('UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt', dtype='U')
ucf101_label = classind[:,1]
fusion_cm = confusion_matrix(video_label,video_fusion_predict)
cm = confusion_matrix(video_label, video_predict)
labels = ucf101_label

np.set_printoptions(precision=2)
fusion_cm_normalized = fusion_cm.astype('float') / fusion_cm.sum(axis=1)[:, np.newaxis]
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_diag = np.diag(cm_normalized)
fusion_cm_diag = np.diag(fusion_cm_normalized)
a = fusion_cm_diag - cm_diag
for i in range(len(labels)):
    print(labels[i],'%.2f%%' % (100 * cm_diag[i]), '%.2f%%' % (100 * fusion_cm_diag[i]),'%.2f%%' % (100*a[i]))
