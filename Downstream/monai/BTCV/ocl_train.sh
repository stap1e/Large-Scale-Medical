now=$(date +"%Y%m%d_%H%M%S")
name=VoCo
pretrained_root=/ai/workspace/pretrained/ocl
logdir=runs/btcv_ocl
feature_size=48
data_dir=/ai/workspace/datasets/VoCo_Downstream/BTCV/
cache_dir=/ai/workspace/datasets/cache/BTCV
use_ssl_pretrained=True
use_persistent_dataset=True

mkdir -p $logdir

python main.py \
    --name $name \
    --batch_size 8 \
    --pretrained_root $pretrained_root \
    --feature_size $feature_size \
    --data_dir $data_dir \
    --cache_dir $cache_dir \
    --use_ssl_pretrained $use_ssl_pretrained \
    --use_persistent_dataset $use_persistent_dataset \
    --logdir $logdir | tee $logdir/$now.txt