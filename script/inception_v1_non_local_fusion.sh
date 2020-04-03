python3.5 train_2d_fusion.py -batch_size 32 -eval_type fusion -model_type inception_v1 -fusion_dr 0.2 -lr_step [20,40] -epoch 60
python3.5 train_2d_fusion.py -batch_size 32 -eval_type fusion -model_type inception_v1 -fusion_dr 0.2 -lr_step [20,40] -epoch 60 -fusion_type non_local

python3.5 test_2d_fusion.py -batch_size 1 -eval_type fusion -model_type inception_v1 -test_crop center
python3.5 test_2d_fusion.py -batch_size 1 -eval_type fusion -model_type inception_v1 -test_crop multi

python3.5 test_2d_fusion.py -batch_size 1 -eval_type fusion -model_type inception_v1 -test_crop center -fusion_type non_local
python3.5 test_2d_fusion.py -batch_size 1 -eval_type fusion -model_type inception_v1 -test_crop multi  -fusion_type non_local
