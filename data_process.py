import os
import sys

def merge_adenoma_records():

    record_file1 = open("/data3/qilei_chen/DATA/polyp_xinzi/preprocessed_4_classification/train.txt")

    record_file2 = open("/data3/qilei_chen/DATA/polyp_xinzi/D2/preprocessed/train.txt")

    record_file3 = open("/data3/qilei_chen/DATA/polyp_xinzi/D1_D2/train.txt","w")

    record = record_file1.readline()

    while record:

        record_file3.write("D1_train/"+record)

        record = record_file1.readline()

    record = record_file2.readline()

    while record:

        record_file3.write("D2_train/"+record)

        record = record_file2.readline()

if __name__ == "__main__":
    merge_adenoma_records()