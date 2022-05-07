export data_dir=/data3/qilei_chen/DATA/polyp_xinzi/D2/preprocessed/
export num_classes=2
export output_dir=/data3/qilei_chen/DATA/polyp_xinzi/D2/preprocessed/work_dir_balanced
export batch_size=64

model_name=swin_base_patch4_window7_224
checkpoint_dir=$output_dir/$model_name-224/model_best.pth.tar
python validate.py $data_dir \
    --model $model_name \
    --num-classes $num_classes \
    --checkpoint $checkpoint_dir \
    -b $batch_size \
    --confusion-matrix \
    --confusion-matrix-fig-dir $output_dir/$model_name-224 \
    --dataset Adenoma --split test