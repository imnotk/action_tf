# python3.5 train_ts_fusion.py -batch_size 32 -eval_type fusion -model_type resnet50 -fusion_dr 0.2 -lr_step [20,40] -epoch 50 -fusion_type non_local -dataset hmdb51 -num_classes 51
# python3.5 test_ts_fusion.py -batch_size 1 -eval_type fusion -model_type resnet50 -fusion_type non_local -test_crop center -frame_counts 25 -gpu_nums 1 -dataset hmdb51 -num_classes 51
# python3.5 test_ts_fusion.py -batch_size 1 -eval_type fusion -model_type resnet50 -fusion_type non_local -test_crop multi -frame_counts 25 -gpu_nums 1 -dataset hmdb51 -num_classes 51

python3.5 train_ts_fusion.py -batch_size 16 -eval_type fusion -model_type resnet50 -fusion_dr 0.2 -lr_step [20,40] -epoch 50 -fusion_type non_local -dataset hmdb51 -num_classes 51 -split 1 -learning_rate 5e-4
python3.5 test_ts_fusion.py -batch_size 1 -eval_type fusion -model_type resnet50 -fusion_type non_local -test_crop center -frame_counts 25 -gpu_nums 1 -dataset hmdb51 -num_classes 51 -split 1
python3.5 test_ts_fusion.py -batch_size 1 -eval_type fusion -model_type resnet50 -fusion_type non_local -test_crop multi -frame_counts 25 -gpu_nums 1 -dataset hmdb51 -num_classes 51 -split 1

python3.5 train_ts_fusion.py -batch_size 16 -eval_type fusion -model_type resnet50 -fusion_dr 0.2 -lr_step [20,40] -epoch 50 -fusion_type non_local -dataset hmdb51 -num_classes 51 -split 2 -learning_rate 5e-4
python3.5 test_ts_fusion.py -batch_size 1 -eval_type fusion -model_type resnet50 -fusion_type non_local -test_crop center -frame_counts 25 -gpu_nums 1 -dataset hmdb51 -num_classes 51 -split 2
python3.5 test_ts_fusion.py -batch_size 1 -eval_type fusion -model_type resnet50 -fusion_type non_local -test_crop multi -frame_counts 25 -gpu_nums 1 -dataset hmdb51 -num_classes 51 -split 2