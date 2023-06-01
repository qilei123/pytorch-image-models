# python inference.py /home/qilei/.TEMP/gastro_position_clasification_11/test/ \
#     --output_dir /home/qilei/.TEMP/gastro_position_clasification_11/work_dir/vit_base_patch32_224-224/ \
#     -m vit_base_patch32_224 \
#     --num-classes 12 \
#     -b 32 \
#     --checkpoint /home/qilei/.TEMP/gastro_position_clasification_11/work_dir/vit_base_patch32_224-224/model_best.pth.tar

python inference.py dataset/FDWJ/FDWJ_SHARES_v4/v4_4/test/ \
    --output_dir dataset/work_dir/v4_4 \
    -m swin_base_patch4_window7_224 \
    --num-classes 4 \
    -b 64 \
    --checkpoint dataset/work_dir/v4_4/swin_base_patch4_window7_224-224/model_best.pth.tar
