import numpy as np
import shutil 
import os

def img2cat(path='ucf101_tvl1_flow/tvl1_flow/u'):
    u_video_path = os.listdir(path)
    for name in u_video_path:
        r_name = os.path.join(path,name.split('_')[1],name)
        o_name = os.path.join(path,name)
        shutil.move(o_name,r_name)

if __name__ == '__main__':
        HandstandPushups = 'HandstandPushups'
        HandStandPushups = 'HandStandPushups'
        
        my_u = "/home/zhujian/video_analysi/zhujian/action_recognition/ucf101_tvl1_flow/tvl1_flow/u/"
        my_v = "/home/zhujian/video_analysi/zhujian/action_recognition/ucf101_tvl1_flow/tvl1_flow/v/"
        
        my_xvid = "/home/zhujian/video_analysi/zhujian/action_recognition/ucf101_flow_video_xvid/HandstandPushups/"
        u_HandstandPushups = os.path.join(my_u,HandstandPushups)
        v_HandstandPushups = os.path.join(my_v,HandstandPushups)
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
        
        for i in os.listdir(my_xvid):
                if HandstandPushups in i:
                        orgin = os.path.join(my_xvid,i)
                        modified_name = i.replace(HandstandPushups,HandStandPushups)
                        modified_path = os.path.join(my_xvid,modified_name)
                        os.rename(orgin,modified_path)