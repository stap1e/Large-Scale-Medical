now=$(date +"%Y%m%d_%H%M%S")
name=VoCo
pretrained_root=/ai/workspace/pretrained/voco
logdir=runs/amos_mri_voco
feature_size=48
data_dir=/ai/workspace/datasets/VoCo_Downstream/Dataset219_AMOS2022_postChallenge_task2/
cache_dir=/ai/workspace/datasets/cache/AMOS_MRI
use_ssl_pretrained=True
use_persistent_dataset=True

mkdir -p $logdir

python main.py \
    --name $name \
    --batch_size 2 \
    --pretrained_root $pretrained_root \
    --feature_size $feature_size \
    --data_dir $data_dir \
    --cache_dir $cache_dir \
    --use_ssl_pretrained $use_ssl_pretrained \
    --use_persistent_dataset $use_persistent_dataset \
    --logdir $logdir | tee $logdir/$now.txt
