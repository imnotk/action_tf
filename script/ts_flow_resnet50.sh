python3.5 train_ts_resnet.py -eval_type flow  -batch_size 16 -learning_rate 5e-4 -split 0 -lr_step [80,160] -epoch 200 -reboot -rgb_dr 1.0 -flow_dr 0.2 -model_type resnet50 -gpu_nums 2   
python3.5 test_ts_resnet.py -eval_type flow -batch_size 1 -model_type resnet50 -video_split 1 -frame_counts 25 -split 0
# python3.5 train_ts_resnet.py -eval_type flow  -batch_size 32 -learning_rate 2e-3 -split 2 -lr_step [150,200] -epoch 250 -reboot -rgb_dr 1.0 -flow_dr 0.2 -model_type resnet50 -gpu_nums 2   
# python3.5 test_ts_resnet.py -eval_type flow -batch_size 1 -model_type resnet50 -video_split 25 -frame_counts 1 -split 2
