export data_dir=/data3/qilei_chen/DATA/polyp_xinzi/preprocessed_4_classification/
export num_classes=12
export output_dir=${data_dir}work_dir
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
    --dataset Adenoma --val-split test