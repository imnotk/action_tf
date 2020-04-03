# python3.5 train_ts_resnet.py -batch_size 16 -eval_type rgb -model_type inception_v1 -learning_rate 5e-4 -lr_step [30,60] -epoch 80 -rgb_dr 1.0 -frame_counts 3 -dataset hmdb51 -num_classes 51
# python3.5 train_ts_resnet.py -batch_size 16 -eval_type rgb -model_type inception_v1 -learning_rate 5e-4 -lr_step [30,60] -epoch 80 -rgb_dr 1.0 -frame_counts 3 -dataset hmdb51 -num_classes 51 -split 1
# python3.5 train_ts_resnet.py -batch_size 16 -eval_type rgb -model_type inception_v1 -learning_rate 5e-4 -lr_step [30,60] -epoch 80 -rgb_dr 1.0 -frame_counts 3 -dataset hmdb51 -num_classes 51 -split 2
python3.5 test_ts_resnet.py -batch_size 1 -eval_type rgb -model_type inception_v1 -split 1  -dataset hmdb51 -num_classes 51
python3.5 test_ts_resnet.py -batch_size 1 -eval_type rgb -model_type inception_v1 -test_crop multi -split 1 -dataset hmdb51 -num_classes 51
python3.5 test_ts_resnet.py -batch_size 1 -eval_type flow -model_type inception_v1 -split 1  -dataset hmdb51 -num_classes 51
python3.5 test_ts_resnet.py -batch_size 1 -eval_type flow -model_type inception_v1 -test_crop multi -split 1 -dataset hmdb51 -num_classes 51
python3.5 test_ts_resnet.py -batch_size 1 -eval_type rgb -model_type inception_v1 -split 2 -dataset hmdb51 -num_classes 51
python3.5 test_ts_resnet.py -batch_size 1 -eval_type rgb -model_type inception_v1 -test_crop multi -split 2 -dataset hmdb51 -num_classes 51
python3.5 test_ts_resnet.py -batch_size 1 -eval_type flow -model_type inception_v1 -split 2 -dataset hmdb51 -num_classes 51
python3.5 test_ts_resnet.py -batch_size 1 -eval_type flow -model_type inception_v1 -test_crop multi -split 2 -dataset hmdb51 -num_classes 51