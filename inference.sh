python inference.py /home/qilei/.TEMP/gastro_position_clasification_11/test/ \
    --output_dir /home/qilei/.TEMP/gastro_position_clasification_11/result/ \
    -m vit_base_patch32_224 \
    --num-classes 12 \
    -b 32 \
    --checkpoint /home/qilei/.TEMP/gastro_position_clasification_11/work_dir/vit_base_patch32_224-224/model_best.pth.tar
