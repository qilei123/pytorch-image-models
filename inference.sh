# python inference.py /home/qilei/.TEMP/gastro_position_clasification_11/test/ \
#     --output_dir /home/qilei/.TEMP/gastro_position_clasification_11/work_dir/vit_base_patch32_224-224/ \
#     -m vit_base_patch32_224 \
#     --num-classes 12 \
#     -b 32 \
#     --checkpoint /home/qilei/.TEMP/gastro_position_clasification_11/work_dir/vit_base_patch32_224-224/model_best.pth.tar

python inference.py /data3/qilei_chen/DATA/polyp_xinzi/D2/preprocessed/test/ \
    --output_dir /data3/qilei_chen/DATA/polyp_xinzi/D2/preprocessed/test/ \
    -m resnext50_32x4d \
    --num-classes 2 \
    -b 32 \
    --checkpoint /data3/qilei_chen/DATA/polyp_xinzi/preprocessed_4_classification/work_dir/resnext50_32x4d-224/model_best.pth.tar
