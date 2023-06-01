# python inference.py /home/qilei/.TEMP/gastro_position_clasification_11/test/ \
#     --output_dir /home/qilei/.TEMP/gastro_position_clasification_11/work_dir/vit_base_patch32_224-224/ \
#     -m vit_base_patch32_224 \
#     --num-classes 12 \
#     -b 32 \
#     --checkpoint /home/qilei/.TEMP/gastro_position_clasification_11/work_dir/vit_base_patch32_224-224/model_best.pth.tar

#-------------D1----------------
# python inference.py /data3/qilei_chen/DATA/polyp_xinzi/retrospective_study_data/ade_positive/ \
#     --output_dir /data3/qilei_chen/DATA/polyp_xinzi/retrospective_study_data/D1/positive/ \
#     -m swin_base_patch4_window7_224 \
#     --num-classes 2 \
#     -b 32 \
#     --checkpoint /data3/qilei_chen/DATA/polyp_xinzi/orig_4_classification/work_dir/swin_base_patch4_window7_224-224/model_best.pth.tar

# python inference.py /data3/qilei_chen/DATA/polyp_xinzi/retrospective_study_data/ade_negative/ \
#     --output_dir /data3/qilei_chen/DATA/polyp_xinzi/retrospective_study_data/D1/negative/ \
#     -m swin_base_patch4_window7_224 \
#     --num-classes 2 \
#     -b 32 \
#     --checkpoint /data3/qilei_chen/DATA/polyp_xinzi/orig_4_classification/work_dir/swin_base_patch4_window7_224-224/model_best.pth.tar
#-------------D2----------------
# python inference.py /data3/qilei_chen/DATA/polyp_xinzi/retrospective_study_data/ade_binary_second_round/ade/ \
#     --output_dir /data3/qilei_chen/DATA/polyp_xinzi/retrospective_study_data/ade_binary_second_round/ade/ \
#     -m swin_base_patch4_window7_224 \
#     --num-classes 2 \
#     -b 32 \
#     --checkpoint /data3/qilei_chen/DATA/polyp_xinzi/D2/preprocessed/work_dir_balanced/swin_base_patch4_window7_224-224/model_best.pth.tar

# python inference.py /data3/qilei_chen/DATA/polyp_xinzi/retrospective_study_data/ade_binary_second_round/none/ \
#     --output_dir /data3/qilei_chen/DATA/polyp_xinzi/retrospective_study_data/ade_binary_second_round/none/ \
#     -m swin_base_patch4_window7_224 \
#     --num-classes 2 \
#     -b 32 \
#     --checkpoint /data3/qilei_chen/DATA/polyp_xinzi/D2/preprocessed/work_dir_balanced/swin_base_patch4_window7_224-224/model_best.pth.tar
#-------------D1_D2----------------
# python inference.py /data3/qilei_chen/DATA/polyp_xinzi/retrospective_study_data/ade_positive/ \
#     --output_dir /data3/qilei_chen/DATA/polyp_xinzi/retrospective_study_data/D1_D2/positive/ \
#     -m swin_base_patch4_window7_224 \
#     --num-classes 2 \
#     -b 32 \
#     --checkpoint /data3/qilei_chen/DATA/polyp_xinzi/D1_D2/work_dir/swin_base_patch4_window7_224-224/model_best.pth.tar

# python inference.py /data3/qilei_chen/DATA/polyp_xinzi/retrospective_study_data/ade_negative/ \
#     --output_dir /data3/qilei_chen/DATA/polyp_xinzi/retrospective_study_data/D1_D2/negative/ \
#     -m swin_base_patch4_window7_224 \
#     --num-classes 2 \
#     -b 32 \
#     --checkpoint /data3/qilei_chen/DATA/polyp_xinzi/D1_D2/work_dir/swin_base_patch4_window7_224-224/model_best.pth.tar

python inference.py /data3/qilei_chen/DATA/301_1600_ade/images_data/回顾性息肉性质判断图片/白光_cropped/ \
    --output_dir /data3/qilei_chen/DATA/301_1600_ade/images_data/回顾性息肉性质判断图片/白光_cropped/D1_D2_result/ \
    -m swin_base_patch4_window7_224 \
    --num-classes 2 \
    -b 32 \
    --checkpoint /data3/qilei_chen/DATA/polyp_xinzi/D1_D2/work_dir/swin_base_patch4_window7_224-224/model_best.pth.tar