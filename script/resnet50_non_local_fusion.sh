

python3.5 train_2d.py -eval_type rgb  -batch_size 16 -learning_rate 5e-4 -split 0 -lr_step [30,60] -epoch 80 -reboot -rgb_dr 1.0 -flow_dr 0.2 -model_type resnet50 -gpu_nums 2
python3.5 test_2d.py -eval_type rgb -batch_size 1 -model_type resnet50 -test_crop multi

python3.5 train_2d_fusion.py -batch_size 32 -eval_type fusion -model_type resnet50 -fusion_dr 0.2 -lr_step [20,40] -epoch 50 -learning_rate 1e-3 -fusion_type non_local

python3.5 test_2d_fusion.py -batch_size 1 -eval_type fusion -model_type resnet50 -fusion_type non_local
python3.5 test_2d_fusion.py -batch_size 1 -eval_type fusion -model_type resnet50 -test_crop multi -fusion_type non_local



# python3.5 train_2d_fusion.py -batch_size 32 -eval_type fusion -model_type resnet101 -fusion_dr 0.2 -lr_step [20,40] -epoch 50 -learning_rate 1e-3 -fusion_type non_local
# python3.5 test_2d_fusion.py -batch_size 1 -eval_type fusion -model_type resnet101
# python3.5 test_2d_fusion.py -batch_size 1 -eval_type fusion -model_type resnet101 -test_crop multi


