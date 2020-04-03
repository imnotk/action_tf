# MIT License
# 
# Copyright (c) 2018 zhujian
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 23:23
# @Author  : zhujian
# @FileName: move_file.py
# @Software: PyCharm

import os,shutil
import numpy as np

# BoxingPunchingBag
# CliffDiving
# GolfSwing
my_classind = "/home/zhujian/video_analysi/zhujian/action_recognition/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt"
# my_classind = r'H:\action_recognition\UCF101TrainTestSplits-RecognitionTask\ucfTrainTestlist\classInd.txt'
# my_classind = r'UCF101TrainTestSplits-RecognitionTask\ucfTrainTestlist\classInd.txt'
my_classind_file = np.genfromtxt(my_classind,dtype='U')
# print(my_classind_file)

my_u = "ucf101_flow_video/"
# my_v = "ucf101_flow_video/v/"
# my_u = r"H:\ucf101_tvl1_flow\tvl1_flow\u"
# my_v = r"H:\ucf101_tvl1_flow\tvl1_flow\v"

# remain_class =['BoxingPunchingBag','CliffDiving','GolfSwing']
# remain_class = list(my_classind_file[:,1])
# for i in my_classind_file:
#     if 'GolfSwing' in i[1]:
#         print(i)
# for i in os.listdir(my_u):
#     print(i)
# for i in remain_class:
#     _classind = i
#     if os.path.exists(os.path.join(my_u,_classind)):
#         print(_classind)
#         continue
#     else:
#         os.makedirs(os.path.join(my_u,_classind))
#         for j in os.listdir(my_u):
#             try:
#                 if _classind == j[2:-12]:
#                     shutil.move(os.path.join(my_u,j),os.path.join(my_u,_classind))
#             except IndexError:
#                 continue

# for i in remain_class:
#     _classind = i
#     if os.path.exists(os.path.join(my_v, _classind)) :
#         print(_classind)
#         continue
#     else :
#         os.makedirs(os.path.join(my_v, _classind))
#         for j in os.listdir(my_v):
#             try:
#                 if _classind == j[2:-12]:
#                     shutil.move(os.path.join(my_v, j), os.path.join(my_v, _classind))
#             except IndexError:
#                 continue

# my_u_list = os.listdir(my_u)
# for i in my_classind_file:
#
#     _classind = i[1]
#     if _classind in remain_class:
#         continue
#     if os.path.exists(os.path.join(my_u,_classind)) and len(os.listdir(os.path.join(my_u,_classind))):
#         print(_classind)
#         continue
#     elif os.path.exists(os.path.join(my_u,_classind)) is False:   os.makedirs(os.path.join(my_u,_classind))
#     for j in my_u_list:
#         if _classind in j:
#             if os.path.exists(os.path.join(my_u,j)):
#                 shutil.move(os.path.join(my_u,j),os.path.join(my_u,_classind,j))
#
# my_v_list = os.listdir(my_v)
# for i in my_classind_file:
#
#     _classind = i[1]
#     if _classind in remain_class:
#         continue
#     if os.path.exists(os.path.join(my_v, _classind)) and len(os.listdir(os.path.join(my_v,_classind))):
#         print(_classind)
#         continue
#     elif os.path.exists (os.path.join (my_v, _classind)) is False: os.makedirs(os.path.join(my_v, _classind))
#     for j in my_v_list:
#         if _classind in j:
#             if os.path.exists(os.path.join(my_v,j)):
#                 shutil.move(os.path.join(my_v, j), os.path.join(my_v, _classind,j))

def img2cat(path='ucf101_tvl1_flow/tvl1_flow/u'):
    u_video_path = os.listdir(path)
    for name in u_video_path:
        r_name = os.path.join(path,name.split('_')[1],name)
        o_name = os.path.join(path,name)
        shutil.move(o_name,r_name)

if __name__ == '__main__':
    img2cat()
    img2cat('ucf101_tvl1_flow/tvl1_flow/v')
HandstandPushups = 'HandstandPushups'
HandStandPushups = 'HandStandPushups'

u_HandstandPushups = os.path.join(my_u,HandstandPushups)
# v_HandstandPushups = os.path.join(my_v,HandstandPushups)
#
for i in os.listdir(u_HandstandPushups):
    if HandstandPushups in i:
        orgin = os.path.join(u_HandstandPushups,i)
        modified_name = i.replace(HandstandPushups,HandStandPushups)
        modified_path = os.path.join(u_HandstandPushups,modified_name)
        os.rename(orgin,modified_path)

for i in os.listdir(v_HandstandPushups):
    if HandstandPushups in i:
        orgin = os.path.join(v_HandstandPushups,i)
        modified_name = i.replace(HandstandPushups,HandStandPushups)
        modified_path = os.path.join(v_HandstandPushups,modified_name)
        os.rename(orgin,modified_path)



# for i in os.listdir(my_u):
#     if i not in my_classind_file[:,1]:
#         shutil.rmtree(os.path.join(my_u,i))
#
# for i in os.listdir(my_v):
#     if i not in my_classind_file[:,1]:
#         shutil.rmtree(os.path.join(my_v,i))
#
#
#
# for i in os.listdir(my_u):
#     dir = os.path.join(my_u,i)
#     if len(os.listdir(dir)) == 0:
#         print(i)
#
# for i in os.listdir(my_v):
#     dir = os.path.join(my_v,i)
#     if len(os.listdir(dir)) == 0:
#         print(i)

