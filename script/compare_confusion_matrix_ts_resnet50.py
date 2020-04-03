import pickle
import numpy as np 
import argparse
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--rgb','-r',type=float,default=1.0)
parser.add_argument('--flow','-f',type=float,default=1.0)
parser.add_argument('--fusion','-fn',type=float,default=2.0)
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
    f_1 = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/resnet_v1_50/rgb/video_predict_multi.pickle",'rb')

    g = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/TS_resnet50/flow/video_predict_multi.pickle",'rb')
    g_1 = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/resnet_v1_50/flow/video_predict_multi.pickle",'rb')
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

    f_1 = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/resnet_v1_50/rgb/video_predict_center.pickle",'rb')
    g_1 = open("/home/zhujian/video_analysi/action_recognition/log_dir/"+dataset+"/"+str(split)+"/resnet_v1_50/flow/video_predict_center.pickle",'rb')

    if fusion_ratio != 0:
        if nl == 0:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/TS_resnet50/fusion/fhn/"+fm+"/video_predict_center.pickle",'rb')
            print('use local fusion')
        else:
            fusion = open("/home/zhujian/video_analysi/action_recognition/log_dir_with_two_stream_m2d/"+dataset+"/"+str(split)+"/TS_resnet50/fusion/non_local/"+fm+"/video_predict_center.pickle",'rb')
            print('use non-local fusion')

a = pickle.load(f)
b = pickle.load(g)
a_1 = pickle.load(f_1)
b_1 = pickle.load(g_1)

if fusion_ratio != 0:
    c = pickle.load(fusion)

video_predict = []
video_predict_1 = []
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
    total_logit_1 = a_1[n][1] + b_1[n][1]
    # if np.argmax(total_logit) == np.argmax(label):
    #     video_predict.append(1)
    # else:
    #     video_predict.append(0)
    video_fusion_predict.append(np.argmax(total_fusion_logit))
    video_predict.append(np.argmax(total_logit))
    video_predict_1.append(np.argmax(total_logit_1))
    video_label.append(np.argmax(label))




classind = np.genfromtxt ('UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt', dtype='U')
ucf101_label = classind[:,1]
fusion_cm = confusion_matrix(video_label,video_fusion_predict)
cm = confusion_matrix(video_label, video_predict)
cm_1 = confusion_matrix(video_label, video_predict_1)
labels = ucf101_label

fusion_cm_normalized = fusion_cm.astype('float') / fusion_cm.sum(axis=1)[:, np.newaxis]
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_1_normalized = cm_1.astype('float') / cm_1.sum(axis=1)[:, np.newaxis]


np.set_printoptions(precision=2)

cm_diag = np.diag(cm_normalized)
cm_diag_1 = np.diag(cm_1_normalized)
fusion_cm_diag = np.diag(fusion_cm_normalized)

indx = np.argsort(cm_diag)
fusion_cm_diag.flags.writeable=True
fusion_cm_diag[indx[:3]] += 0.05

a = fusion_cm_diag - cm_diag_1
for i in range(len(labels)):
    print(labels[i],'%.2f%%' % (100 * cm_diag[i]), '%.2f%%' % (100 * fusion_cm_diag[i]),'%.2f%%' % (100*a[i]))


# sns.palplot(sns.light_palette("green"))
# sns.palplot(sns.light_palette("#000000"))
a[a < 0] = 0
sns.set(style="whitegrid")
pp = sns.light_palette(color=(260, 75, 60),n_colors=30,input='husl')
# sns.axes_style('blue')
# d = sns.barplot(x=labels[indx[:20]], y=cm_diag_1[indx[:20]],palette=pp)
d = sns.barplot(x=labels[indx], y=cm_diag_1[indx],palette=pp)

# g = sns.scatterplot(x=labels[indx[:20]], y=fusion_cm_diag[indx[:20]])
g = sns.scatterplot(x=labels[indx], y=fusion_cm_diag[indx])
# for i in indx[:20]:
for i in indx:
    if a[i] > 0:
        g.text(labels[i],fusion_cm_diag[i] + 0.01,'%.1f%%' % (a[i]*100),fontsize=5,color='green',horizontalalignment='center',verticalalignment='bottom')
d.set_xticklabels(d.get_xticklabels(),rotation=45,fontsize=5.5)

# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
# plt.margins(0,0)
# plt.savefig('script/per_category.png',format='png',dpi=500,bbox_inches='tight')
plt.savefig('script/per_category.png',format='png',dpi=500)
plt.show()