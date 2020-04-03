# python3.5 train_ts_resnet.py -eval_type rgb  -batch_size 16 -learning_rate 5e-4 -split 0 -lr_step [25,50] -epoch 60 -reboot -rgb_dr 1.0 -flow_dr 0.2 -model_type resnet50 -gpu_nums 2   
# python3.5 test_ts_resnet.py -eval_type rgb -batch_size 1 -model_type resnet50 -video_split 25 -frame_counts 1
python3.5 train_ts_resnet.py -eval_type rgb  -batch_size 16 -learning_rate 5e-4 -split 1 -lr_step [30,60] -epoch 80 -reboot -rgb_dr 1.0 -flow_dr 0.2 -model_type resnet50 -gpu_nums 2   
python3.5 test_ts_resnet.py -eval_type rgb -batch_size 1 -model_type resnet50 -video_split 25 -frame_counts 1 -split 1
python3.5 train_ts_resnet.py -eval_type rgb  -batch_size 16 -learning_rate 5e-4 -split 2 -lr_step [30,60] -epoch 80 -reboot -rgb_dr 1.0 -flow_dr 0.2 -model_type resnet50 -gpu_nums 2   
python3.5 test_ts_resnet.py -eval_type rgb -batch_size 1 -model_type resnet50 -video_split 25 -frame_counts 1 -split 2
