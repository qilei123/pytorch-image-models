root_dir=/data3/qilei_chen/DATA/polyp_xinzi/preprocessed_4_classification
output_dir=$root_dir/work_dir
num_class=2
# python train.py $root_dir --model mobilenetv3_large_100 --pretrained --num-classes $num_class --output $output_dir -b 64 --dataset Adenoma --val-split test
# python train.py $root_dir --model densenet121 --pretrained --num-classes $num_class --output $output_dir -b 64 --dataset Adenoma --val-split test
# python train.py $root_dir --model resnet50 --pretrained --num-classes $num_class --output $output_dir -b 64 --dataset Adenoma --val-split test
# python train.py $root_dir --model efficientnet_b0 --pretrained --num-classes $num_class --output $output_dir -b 64 --dataset Adenoma --val-split test
#transformer
# python train.py $root_dir --model vit_base_patch32_224 --pretrained --num-classes $num_class --output $output_dir -b 64 --dataset Adenoma --val-split test
# python train.py $root_dir --model twins_pcpvt_base --pretrained --num-classes $num_class --output $output_dir -b 64 --dataset Adenoma --val-split test
# python train.py $root_dir --model swin_base_patch4_window7_224 --pretrained --num-classes $num_class --output $output_dir -b 64 --dataset Adenoma --val-split test
# python train.py $root_dir --model convit_base --pretrained --num-classes $num_class --output $output_dir -b 64 --dataset Adenoma --val-split test

python train.py $root_dir --model resnext50_32x4d --pretrained --num-classes $num_class --output $output_dir -b 64 --dataset Adenoma --val-split test
