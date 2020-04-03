import cv2
import numpy as np
import os
import glob
import skvideo.io

flow_u_path=r'E:/UCF101\ucf101_tvl1_flow/tvl1_flow/u'
flow_v_path=r'E:/UCF101/ucf101_tvl1_flow/tvl1_flow/v'
rgb_path = r'E:\UCF101\ucf101_jpegs_256\jpegs_256'

# flow_u_path = r'E:\hmdb51\hmdb51_tvl1_flow\tvl1_flow\u'
# flow_v_path = r'E:\hmdb51\hmdb51_tvl1_flow\tvl1_flow\v'
# rgb_path = r'E:\hmdb51\hmdb51_jpegs_256\jpegs_256'

# flow_u_path="/mnt/zhujian/ucf101_flow/ucf101_tvl1_flow/tvl1_flow/u/"
# flow_v_path="/mnt/zhujian/ucf101_flow/ucf101_tvl1_flow/tvl1_flow/v/"
# rgb_path = "/mnt/zhujian/jpegs_256"

save_u_path = r'E:/UCF101/tvl1_flow_video/u'
save_v_path = r'E:/UCF101/tvl1_flow_video/v'
save_rgb_path = r'E:/UCF101/ucf_rgb_video'

# save_u_path = r'E:\hmdb51\tvl1_flow_video\u'
# save_v_path = r'E:\hmdb51\tvl1_flow_video\v'
# save_rgb_path = r'E:/hmdb51/ucf_rgb_video'

# save_u_path="/mnt/zhujian/ucf101_flow_video/u/"
# save_v_path="/mnt/zhujian/ucf101_flow_video/v/"
# save_rgb_path = "/mnt/zhujian/ucf101_rgb_video"

def imgs2video(imgs_dir, save_name):

    fps = 25
    # print(save_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (340, 256))
    # if os.path.exists(save_name):
    #     return
    if video_writer.isOpened() is False:
        raise ValueError('video writer not opened',imgs_dir)

    # no glob, need number-index increasing
    imgs = glob.glob(os.path.join(imgs_dir, '*.jpg'))
    # print(imgs)
    for i in range(len(imgs)):
        imgname = imgs[i]
        frame = cv2.imread(imgname)
        if frame is None:
            print(imgname)
            continue
        frame = cv2.resize(frame,(340,256))
        video_writer.write(frame)

    video_writer.release()

def flow_imgs2video(flow_path,save_path):
    img_dir = os.listdir(flow_path)
    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)
    for i in img_dir:
        if i.endswith('.bin'):
            continue
        i_dir = os.path.join(flow_path,i)
        s_name = os.path.join(save_path,i+'.avi')
        imgs2video(i_dir,s_name)
        print(s_name)

def two_video2_one_video(u_path='/mnt/zhujian/ucf101_flow_video/u',
                         v_path='/mnt/zhujian/ucf101_flow_video/v',
                         save_path='/mnt/zhujian/final_ucf101_flow_video'):
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for name in os.listdir(u_path):
        u_video = os.path.join(u_path,name)
        v_video = os.path.join(v_path,name)
        cap_u = cv2.VideoCapture(u_video)
        cap_v = cv2.VideoCapture(v_video)
        fps = 25
        save_name = os.path.join(save_path,name)
        fourcc = cv2.VideoWriter_fourcc (*'MJPG')
        # fourcc = cap_u.get(cv2.CAP_PROP_FOURCC)
        video_writer = cv2.VideoWriter (save_name, fourcc, fps, (340, 256))
        video_writer.set(cv2.CAP_PROP_FPS,25)
        if video_writer.isOpened() is False:
            raise ValueError('video writer not opened',save_name)
        if cap_u.get(cv2.CAP_PROP_FRAME_COUNT) != cap_v.get(cv2.CAP_PROP_FRAME_COUNT):
            print('frame count is not equal')
        img_list = []
        while 1:
            flag_u,img_u = cap_u.read()
            flag_v,img_v = cap_v.read()
            if flag_u is False or img_u is None:
                break
            img_u = cv2.cvtColor(img_u,cv2.COLOR_BGR2GRAY)
            img_v = cv2.cvtColor(img_v,cv2.COLOR_BGR2GRAY)
            img_u = cv2.resize (img_u, (340, 256))
            img_v = cv2.resize (img_v, (340, 256))
            img_final = np.zeros(img_u.shape)
            img = np.stack([img_u,img_v,img_final],axis=-1)
            img = cv2.resize (img, (340, 256))
            img = np.uint8(img)
            img_list.append(img)
        for i in img_list:
            video_writer.write(i)
        print(save_name,video_writer.get(cv2.CAP_PROP_FRAME_COUNT))
        video_writer.release ()



if __name__ == '__main__':
    # flow_imgs2video(flow_u_path,save_u_path)
    # flow_imgs2video(flow_v_path,save_v_path)
    # imgs2video(r'E:\UCF101\ucf101_tvl1_flow\tvl1_flow\v\v_PommelHorse_g05_c01', r'E:/UCF101/tvl1_flow_video/v/v_PommelHorse_g05_c01.avi')
    # imgs2video(r'E:\UCF101\ucf101_tvl1_flow\tvl1_flow\v\v_PommelHorse_g05_c02', r'E:/UCF101/tvl1_flow_video/v/v_PommelHorse_g05_c02.avi')
    # imgs2video(r'E:\UCF101\ucf101_tvl1_flow\tvl1_flow\v\v_PommelHorse_g05_c03', r'E:/UCF101/tvl1_flow_video/v/v_PommelHorse_g05_c03.avi')
    # imgs2video(r'E:\UCF101\ucf101_tvl1_flow\tvl1_flow\v\v_PommelHorse_g05_c04', r'E:/UCF101/tvl1_flow_video/v/v_PommelHorse_g05_c04.avi')
    # flow_imgs2video(rgb_path,save_rgb_path)
    two_video2_one_video(save_u_path,save_v_path)