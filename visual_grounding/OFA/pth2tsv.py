import csv        
import torch
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathfrom', default=None)
    parser.add_argument('--pathto', default=None)
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    name_list = ['ENV1_train', 'ENV2_train']
    if args.name:
        name_list = []
        name_list.append(args.name)
    for name in name_list:

        pseudo_sample_path = '../../instruction_generation/data/pseudo_samples/{}/{}.pth'.format(name,name)
        if args.pathfrom:
            pseudo_sample_path = args.pathfrom
            
        tsv_path = '../../data/train'
        if args.pathto:
            tsv_path = args.pathto

        print(pseudo_sample_path)
        print(tsv_path)

        data = torch.load(pseudo_sample_path)
        print('number of data in {} = {}'.format(name, len(data)))

        with open(os.path.join(tsv_path,'{}.tsv'.format(name)), 'w', newline='') as f_output:
            tsv_output = csv.writer(f_output, delimiter='\t')
            for indx, sample in enumerate(data):
                use = []
                file_path = os.path.join(tsv_path,'{}'.format(name))
                if '\n' in sample[0]:
                    file_name = sample[0][:-1]
                else:
                    file_name = sample[0]
                use.append(indx)
                use.append(sample[0].split('.')[0])
                use.append(sample[3])
                use.append("{},{},{},{}".format(sample[2][0],sample[2][1],sample[2][2],sample[2][3]))
                img = Image.open(file_path + file_name)
                img_buffer = BytesIO()
                img.save(img_buffer, format=img.format)
                byte_data = img_buffer.getvalue()
                base64_str = base64.b64encode(byte_data) # bytes
                base64_str = base64_str.decode("utf-8") # str
                use.append(base64_str)
                tsv_output.writerow(use)
                if indx%1000 == 0:
                    print("{} directory processing ... {}%".format(name, int(100*indx/len(data))))