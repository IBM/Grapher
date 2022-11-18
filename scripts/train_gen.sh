#!/usr/bin/env bash
cd ../
python main.py    --version 2\
                  --default_root_dir output \
                  --run train \
                  --max_epochs 100 \
                  --accelerator gpu \
                  --num_nodes 1 \
                  --num_data_workers 4 \
                  --lr 1e-4 \
                  --batch_size 11 \
                  --num_sanity_val_steps 0 \
                  --fast_dev_run 0 \
                  --overfit_batches 0 \
                  --limit_train_batches 1.0 \
                  --limit_val_batches 1.0 \
                  --limit_test_batches 1.0 \
                  --accumulate_grad_batches 10 \
                  --detect_anomaly True \
                  --data_path webnlg-dataset/release_v3.0/en \
                  --log_every_n_steps 100 \
                  --val_check_interval 1000 \
                  --checkpoint_step_frequency 1000 \
                  --focal_loss_gamma 3 \
                  --dropout_rate 0.5 \
                  --num_layers 2 \
                  --edges_as_classes 0 \
                  --checkpoint_model_id -1 \
