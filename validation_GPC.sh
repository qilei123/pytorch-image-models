export data_dir=/home/qilei/.TEMP/gastro_position_clasification/
export num_classes=12
#export output_dir=${data_dir}work_dir_balanced/
export output_dir=${data_dir}work_dir/
export batch_size=16

#model_name=vit_base_patch32_224
#python train.py $data_dir --model $model_name --pretrained --num-classes $num_classes --output $output_dir -b $batch_size --use_balanced_sampler
#model_name=twins_pcpvt_base
#python train.py $data_dir --model $model_name --pretrained --num-classes $num_classes --output $output_dir -b $batch_size --use_balanced_sampler
model_name=xcit_small_12_p8_224
checkpoint_dir= $output_dir/$model_name-224/model_best.pth.tar
python validate.py $data_dir --model $model_name --num-classes $num_classes --checkpoint $checkpoint_dir -b $batch_size --confusion-matrix --confusion-matrix-fig-dir $output_dir/$model_name-224
#model_name=swin_base_patch4_window7_224
#python train.py $data_dir --model $model_name --pretrained --num-classes $num_classes --output $output_dir -b $batch_size --use_balanced_sampler
