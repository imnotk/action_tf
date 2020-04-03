sleep 7777
# python3.5 train_2d_fusion.py -batch_size 32 -eval_type fusion -model_type resnet101 -fusion_dr 0.2 -lr_step [25,50] -epoch 70 -learning_rate 1e-3 -fusion_type non_local
python3.5 test_2d_fusion.py -batch_size 1 -eval_type fusion -model_type resnet101 -fusion_type non_local -test_crop center 
python3.5 test_2d_fusion.py -batch_size 1 -eval_type fusion -model_type resnet101 -fusion_type non_local -test_crop multi