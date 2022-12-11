export data_dir=/home/qilei/.TEMP/gastro_position_clasification_best/
export num_classes=13
#export output_dir=${data_dir}work_dir_balanced/
export output_dir=${data_dir}work_dir/
export batch_size=32

model_name=vit_base_patch32_224
python train.py $data_dir --model $model_name --pretrained --num-classes $num_classes --output $output_dir -b $batch_size
model_name=twins_pcpvt_base
python train.py $data_dir --model $model_name --pretrained --num-classes $num_classes --output $output_dir -b $batch_size
model_name=xcit_small_12_p8_224
python train.py $data_dir --model $model_name --pretrained --num-classes $num_classes --output $output_dir -b $batch_size
model_name=swin_base_patch4_window7_224
python train.py $data_dir --model $model_name --pretrained --num-classes $num_classes --output $output_dir -b $batch_size
