import os

name = 'ENV1_train' # image directory
image_path = '../data/train/'+name

L = os.listdir(image_path)
L.sort()
print(len(L))

save_file_path = '../instruction_generation/data/statistic/{}/{}_train_imagelist_split0.txt'.format(name, name)
save_file_path2 = './{}_train_imagelist_split0.txt'.format(name)

with open(save_file_path, 'w+') as lf:
    lf.write('\n'.join(L))

with open(save_file_path2, 'w+') as lf:
    lf.write('\n'.join(L))