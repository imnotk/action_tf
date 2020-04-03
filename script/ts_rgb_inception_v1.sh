# python3.5 train_ts_resnet.py -batch_size 32 -eval_type rgb -model_type inception_v1 -learning_rate 1e-3 -lr_step [30,60] -epoch 80 -rgb_dr 0.2 -frame_counts 7
python3.5 test_ts_resnet.py -batch_size 1 -eval_type rgb -model_type inception_v1 -split 1 
python3.5 test_ts_resnet.py -batch_size 1 -eval_type rgb -model_type inception_v1 -test_crop multi -split 1
python3.5 test_ts_resnet.py -batch_size 1 -eval_type flow -model_type inception_v1 -split 1 
python3.5 test_ts_resnet.py -batch_size 1 -eval_type flow -model_type inception_v1 -test_crop multi -split 1
python3.5 test_ts_resnet.py -batch_size 1 -eval_type rgb -model_type inception_v1 -split 2
python3.5 test_ts_resnet.py -batch_size 1 -eval_type rgb -model_type inception_v1 -test_crop multi -split 2
python3.5 test_ts_resnet.py -batch_size 1 -eval_type flow -model_type inception_v1 -split 2
python3.5 test_ts_resnet.py -batch_size 1 -eval_type flow -model_type inception_v1 -test_crop multi -split 2
