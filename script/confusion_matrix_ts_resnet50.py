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
    f = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/TS_resnet50/rgb/video_predict_multi.pickle",'rb')
    g = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/TS_resnet50/flow/video_predict_multi.pickle",'rb')
    print('use ten crop')
    if fusion_ratio != 0:
        if nl == 0:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/TS_resnet50/fusion/fhn/"+fm+"/video_predict_multi.pickle",'rb')
            print('use local fusion')
        else:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/TS_resnet50/fusion/non_local/"+fm+"/video_predict_multi.pickle",'rb')
            print('use non-local fusion')
else:
    print('use center crop')
    f = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/TS_resnet50/rgb/video_predict_center.pickle",'rb')
    g = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/TS_resnet50/flow/video_predict_center.pickle",'rb')
    if fusion_ratio != 0:
        if nl == 0:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/TS_resnet50/fusion/fhn/"+fm+"/video_predict_center.pickle",'rb')
            print('use local fusion')
        else:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/TS_resnet50/fusion/non_local/"+fm+"/video_predict_center.pickle",'rb')
            print('use non-local fusion')

a = pickle.load(f)
b = pickle.load(g)
if fusion_ratio != 0:
    c = pickle.load(fusion)

video_predict = []
video_softmax_predict = []
video_label = []
for n , v in enumerate(a):
    label = a[n][0]
    rgb_logit = a[n][1]
    flow_logit = b[n][1]
    if fusion_ratio != 0:
        fusion_logit = c[n][1]
        total_logit = r_ratio * rgb_logit +  f_ratio * flow_logit + fusion_ratio * fusion_logit
    else:
        total_logit = r_ratio * rgb_logit +  f_ratio * flow_logit
    # if np.argmax(total_logit) == np.argmax(label):
    #     video_predict.append(1)
    # else:
    #     video_predict.append(0)
    video_predict.append(np.argmax(total_logit))
    video_label.append(np.argmax(label))




classind = np.genfromtxt ('UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt', dtype='U')
ucf101_label = classind[:,1]
print(ucf101_label)
cm = confusion_matrix(video_label, video_predict)
labels = ucf101_label

np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_diag = np.diag(cm_normalized)
worst = np.argmin(cm_diag)
print(cm_normalized)
print(cm_diag)
print(worst, labels[worst])
print(sorted(list(zip(cm_normalized[worst],labels))))
plt.figure(figsize=(12, 8), dpi=120)
plt.rcParams['figure.dpi'] = 300


ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)
tick_marks = np.array(range(len(labels))) + 0.5
# for x_val, y_val in zip(x.flatten(), y.flatten()):
#     c = cm_normalized[y_val][x_val]
#     if c > 0.01:
#         plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90, fontsize=5)
    plt.yticks(xlocations, labels,fontsize=5)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix',cmap=plt.cm.Spectral)
# show confusion matrix
plt.savefig('script/confusion_matrix.png', format='png', dpi=500,bbox_inches='tight')
# plt.show()
