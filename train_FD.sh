root_dir=/home/qilei/.TEMP/放大胃镜图片筛选/v3_3

num_class=3

output_dir=$root_dir/work_dir
dataset=folder_fdv3

python train.py $root_dir --model resnext50_32x4d --pretrained --num-classes $num_class --output $output_dir -b 64 --dataset $dataset --val-split test --epochs 100 #-j 1
#transformer
python train.py $root_dir --model swin_base_patch4_window7_224 --pretrained --num-classes $num_class --output $output_dir -b 32 --dataset $dataset --val-split test --epochs 100 #-j 1

output_dir=$root_dir/work_dir_balance
python train.py $root_dir --model resnext50_32x4d --pretrained --num-classes $num_class --output $output_dir -b 64 --dataset $dataset --val-split test --epochs 100 --use_balanced_sampler #-j 1
#transformer
python train.py $root_dir --model swin_base_patch4_window7_224 --pretrained --num-classes $num_class --output $output_dir -b 32 --dataset $dataset --val-split test --epochs 100 --use_balanced_sampler #-j 1


#python train.py E:/DATASET/放大胃镜/放大胃镜图片筛选 --model resnext50_32x4d --pretrained --num-classes 4 --output E:/DATASET/放大胃镜/放大胃镜图片筛选/work_dir -b 16 --dataset FDV1 --val-split test