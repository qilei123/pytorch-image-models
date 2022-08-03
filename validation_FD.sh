data_dir=/home/qilei/.TEMP/放大胃镜图片筛选/v3/
num_classes=4
#export output_dir=${data_dir}work_dir_balanced
output_dir=${data_dir}work_dir
batch_size=64

model_name=resnext50_32x4d
checkpoint_dir=$output_dir/$model_name-224/model_best.pth.tar
python validate.py ${data_dir}test --model $model_name --num-classes $num_classes --checkpoint $checkpoint_dir -b $batch_size --confusion-matrix --confusion-matrix-fig-dir $output_dir/$model_name-224 -j 8 --results-file output_dir/$model_name-224/results.cvs

model_name=swin_base_patch4_window7_224
checkpoint_dir=$output_dir/$model_name-224/model_best.pth.tar
python validate.py ${data_dir}test --model $model_name --num-classes $num_classes --checkpoint $checkpoint_dir -b $batch_size --confusion-matrix --confusion-matrix-fig-dir $output_dir/$model_name-224 -j 8 --results-file output_dir/$model_name-224/results.cvs

output_dir=${data_dir}work_dir_balance

model_name=resnext50_32x4d
checkpoint_dir=$output_dir/$model_name-224/model_best.pth.tar
python validate.py ${data_dir}test --model $model_name --num-classes $num_classes --checkpoint $checkpoint_dir -b $batch_size --confusion-matrix --confusion-matrix-fig-dir $output_dir/$model_name-224 -j 8 --results-file output_dir/$model_name-224/results.cvs

model_name=swin_base_patch4_window7_224
checkpoint_dir=$output_dir/$model_name-224/model_best.pth.tar
python validate.py ${data_dir}test --model $model_name --num-classes $num_classes --checkpoint $checkpoint_dir -b $batch_size --confusion-matrix --confusion-matrix-fig-dir $output_dir/$model_name-224 -j 8 --results-file output_dir/$model_name-224/results.cvs

#python validate.py /home/qilei/.TEMP/放大胃镜图片筛选/v1/ --model resnext50_32x4d --num-classes 4 --checkpoint /home/qilei/.TEMP/放大胃镜图片筛选/v1/work_dir_balance/resnext50_32x4d-224/model_best.pth.tar -b 32 --confusion-matrix --confusion-matrix-fig-dir /home/qilei/.TEMP/放大胃镜图片筛选/v1/work_dir_balance/resnext50_32x4d-224