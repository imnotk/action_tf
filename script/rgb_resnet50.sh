
python3.5 train_2d.py -eval_type rgb  -batch_size 16 -learning_rate 5e-4 -split 0 -lr_step [30,60] -epoch 80 -reboot -rgb_dr 1.0 -flow_dr 0.2 -model_type resnet50 -gpu_nums 2
python3.5 train_2d.py -eval_type rgb  -batch_size 16 -learning_rate 5e-4 -split 0 -lr_step [30,60] -epoch 80 -reboot -rgb_dr 1.0 -flow_dr 0.2 -model_type resnet101 -gpu_nums 2   

python3.5 test_2d.py -eval_type rgb -batch_size 1 -model_type resnet50
python3.5 test_2d.py -eval_type rgb -batch_size 1 -model_type resnet50 -test_crop multi

python3.5 test_2d.py -eval_type rgb -batch_size 1 -model_type resnet101
python3.5 test_2d.py -eval_type rgb -batch_size 1 -model_type resnet101 -test_crop multi

python3.5 train_ts_resnet.py -eval_type rgb -batch_size 16 -learning_rate 5e-4 -split 0 -lr_step [30,60] -epoch 80 -reboot -rgb_dr 1.0 -flow_dr 0.2 -model_type inception_v1 -gpu_nums 2