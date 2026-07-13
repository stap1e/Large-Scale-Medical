now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_ocl_swin_B
mkdir -p $logdir

torchrun --master_port=28814 ocl_train.py \
    --logdir $logdir | tee $logdir/$now.txt
