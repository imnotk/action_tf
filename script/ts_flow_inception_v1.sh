sleep 7200
python3.5 train_ts_resnet.py -batch_size 32 -eval_type flow -model_type inception_v1 -learning_rate 2e-3 -lr_step [80,160] -epoch 200 -flow_dr 0.2 -frame_counts 3 