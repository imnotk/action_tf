import os

u_path = "/home/zhujian/video_analysi/zhujian/action_recognition/ucf101_tvl1_flow/tvl1_flow/u/"
# u_path = "/home/zhujian/video_analysi/zhujian/action_recognition/hmdb51_tvl1_flow/u/"
title = os.listdir(u_path)

min_len = 100000
s = ''
for i in title:
    t_path = os.path.join(u_path,i)
    for j in os.listdir(t_path):
        i_path = os.path.join(t_path,j)
        l  = len(os.listdir(i_path))
        if l < min_len:
            s = i_path
            min_len = l

print(s,min_len)