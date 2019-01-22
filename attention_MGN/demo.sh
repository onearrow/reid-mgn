# train-lwp
# 2018.12.4 11：00->18：30
# mAP: 0.9418 rank1: 0.9587 rank3: 0.9748 rank5: 0.9798 rank10: 0.9843 (Best: 0.9418 @epoch 400)
# CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/mgn/Market-1501-v15.09.15/ --batchid 8 --batchtest 16 --test_every 50 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save lwp-2018.12.4 --nGPU 1 --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad

# 2018.12.10 9:00->17:36, no rerank
# mAP: 0.8789 rank1: 0.9489 rank3: 0.9739 rank5: 0.9810 rank10: 0.9887 (Best: 0.8789 @epoch 400)
# CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/mgn/Market-1501-v15.09.15/ --batchid 8 --batchtest 16 --test_every 50 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save lwp-2018.12.7 --nGPU 1 --lr 2e-4 --optimizer ADAM --random_erasing --reset --amsgrad

# 2018.12.11 23:16->?, PReLU
# CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/mgn/Market-1501-v15.09.15/ --batchid 8 --batchtest 16 --test_every 50 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save lwp-2018.12.11 --nGPU 1 --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad
# mAP: 0.9472 rank1: 0.9614 rank3: 0.9780 rank5: 0.9816 rank10: 0.9875 (Best: 0.9472 @epoch 350)

# 2018.12.12 11:50-> , PReLU
# CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/mgn/Market-1501-v15.09.15/ --batchid 8 --batchtest 16 --test_every 10 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save lwp-2018.12.12 --nGPU 1 --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad
# mAP: 0.9468 rank1: 0.9626 rank3: 0.9757 rank5: 0.9789 rank10: 0.9860 (Best: 0.9469 @epoch 390)

# 2018.12.19 17:05-> , PReLU
# CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/mgn/Market-1501-v15.09.15/ --batchid 8 --batchtest 16 --test_every 10 --epochs 20 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save lwp-2018.12.19 --nGPU 1 --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad

# test-lwp
# python main.py --datadir /home/lwp/beednprojects/mgn/Market-1501-v15.09.15/ --margin 1.2 --save lwp-test --cpu --test_only --resume 0 --pre_train model/model_best.pt --batchtest 16 --re_rank
# CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/Market-1501-v15.09.15/ --margin 1.2 --save MGN-test --nGPU 1 --test_only --resume 0 --pre_train model/model_12_27.pt --batchtest 16 --re_rank
# CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/Market-1501-v15.09.15/ --margin 1.2 --save MGN_01_11_M_C-test --nGPU 1 --test_only --resume 0 --pre_train model/model_01_11_M_C.pt --batchtest 16 --re_rank --num_classes 749
# CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/re-id_data/homedepot_2018-09-14/re-id_H/ --margin 1.2 --save lwp-test_2018.12.25_H --nGPU 1 --test_only --resume 0 --pre_train model/model_latest.pt --batchtest 16 --re_rank
# CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/Market-1501-v15.09.15/ --margin 1.2 --save MGN_01_11-test --nGPU 1 --test_only --resume 0 --pre_train model/model_01_11.pt --batchtest 16 --re_rank --num_classes 963


# 2018.12.25 18:27-> , 
# CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/mgn/Market-1501-v15.09.15/ --batchid 8 --batchtest 16 --test_every 20 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save lwp-2018.12.27 --nGPU 1 --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad
# mAP: 0.9471 rank1: 0.9623 rank3: 0.9751 rank5: 0.9795 rank10: 0.9855 (Best: 0.9471 @epoch 360)

# 2019.01.03 16:30-> , 
# CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/mgn/Market-1501-v15.09.15_cleaned/ --batchid 8 --batchtest 16 --test_every 20 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save lwp-2019.01.03 --nGPU 1 --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad --num_classes 749
# mAP: 0.9787 rank1: 0.9469 rank3: 0.9667 rank5: 0.9718 rank10: 0.9798 (Best: 0.9787 @epoch 140)

# 2019.01.11 09:52-> , 
# CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/mgn/Market-1501-v15.09.15_cleaned/ --batchid 8 --batchtest 16 --test_every 20 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save lwp-2019.01.11 --nGPU 1 --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad --num_classes 749

# 2019.01.22 16:16-> , 
CUDA_VISIBLE_DEVICES=0 python main.py --datadir /home/lwp/beednprojects/mgn/Market-1501-v15.09.15/ --batchid 8 --batchtest 16 --test_every 20 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save lwp-01_21 --nGPU 1 --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad
