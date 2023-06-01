export data_dir=dataset/FDWJ/FDWJ_SHARES_v4/v4_3/test
export num_classes=3
#export output_dir=${data_dir}work_dir_balanced
export output_dir=dataset/work_dir/v4_3
export batch_size=64

model_name=resnext50_32x4d
checkpoint_dir=$output_dir/$model_name-224/model_best.pth.tar
python validate.py $data_dir --model $model_name --num-classes $num_classes --checkpoint $checkpoint_dir -b $batch_size --confusion-matrix --confusion-matrix-fig-dir $output_dir/$model_name-224

model_name=swin_base_patch4_window7_224
checkpoint_dir=$output_dir/$model_name-224/model_best.pth.tar
python validate.py $data_dir --model $model_name --num-classes $num_classes --checkpoint $checkpoint_dir -b $batch_size --confusion-matrix --confusion-matrix-fig-dir $output_dir/$model_name-224
