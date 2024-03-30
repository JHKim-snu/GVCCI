##############################################
# need images/pnp_test directory!!!

import sys
import socket
import cv2
import numpy as np
import base64

##############################################
###############vvvvvvvvvvvvvvvv################
import torch
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.refcoco import RefcocoTask
from models.ofa import OFAModel
from PIL import Image
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('--model', help="model time step e.g. 0,8,33,135,540")
#parser.add_argument('--query', help="instruction number e.g. 0,1,2,...,59")
#args = parser.parse_args()

rel_list = [
    'on the ',
    'right side of the ',
    'left side of the ',
    'in front of the ',
    'behind of the '
]

tasks.register_task('refcoco', RefcocoTask)
use_cuda = torch.cuda.is_available()
use_fp16 = False
overrides={"bpe_dir":"utils/BPE"}

our_pick_model_path = '' # INSERT your model directory
# INSERT your model checkpoint
pick_model_path_dict = {
    '135': our_pick_model_path + '0208_train_135/checkpoint_last.pt',
}

our_place_model_path = '' # INSERT your model directory
# INSERT your model checkpoint
place_model_path_dict = {
    '135': our_place_model_path + 'place_0208_train_135/checkpoint_last.pt',

}

#load pick models, cfg, task
for model_num, model_path in pick_model_path_dict.items():
    globals()['pick_models_{}'.format(model_num)], globals()['pick_cfg_{}'.format(model_num)], globals()['pick_task_{}'.format(model_num)] = \
        checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(model_path),
        arg_overrides=overrides
    )
    globals()['pick_cfg_{}'.format(model_num)].common.seed = 7
    globals()['pick_cfg_{}'.format(model_num)].generation.beam = 5
    globals()['pick_cfg_{}'.format(model_num)].generation.min_len = 4
    globals()['pick_cfg_{}'.format(model_num)].generation.max_len_a = 0
    globals()['pick_cfg_{}'.format(model_num)].generation.max_len_b = 4
    globals()['pick_cfg_{}'.format(model_num)].generation.no_repeat_ngram_size = 3

    
    # Move models to GPU
    for globals()['pick_model_{}'.format(model_num)] in globals()['pick_models_{}'.format(model_num)]:
        globals()['pick_model_{}'.format(model_num)].eval()
        if use_fp16:
            globals()['pick_model_{}'.format(model_num)].half()
        if use_cuda and not globals()['pick_cfg_{}'.format(model_num)].distributed_training.pipeline_model_parallel:
            globals()['pick_model_{}'.format(model_num)].cuda()
        globals()['pick_model_{}'.format(model_num)].prepare_for_inference_(globals()['pick_cfg_{}'.format(model_num)])
        print("pick model {} on the device".format(model_num))

    # Initialize generator
    globals()['pick_generator_{}'.format(model_num)] = globals()['pick_task_{}'.format(model_num)].build_generator(globals()['pick_models_{}'.format(model_num)], globals()['pick_cfg_{}'.format(model_num)].generation)


# load place models, cfg, task
for model_num, model_path in place_model_path_dict.items():
    globals()['place_models_{}'.format(model_num)], globals()['place_cfg_{}'.format(model_num)], globals()['place_task_{}'.format(model_num)] = \
        checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(model_path),
        arg_overrides=overrides
    )
    globals()['place_cfg_{}'.format(model_num)].common.seed = 7
    globals()['place_cfg_{}'.format(model_num)].generation.beam = 5
    globals()['place_cfg_{}'.format(model_num)].generation.min_len = 4
    globals()['place_cfg_{}'.format(model_num)].generation.max_len_a = 0
    globals()['place_cfg_{}'.format(model_num)].generation.max_len_b = 4
    globals()['place_cfg_{}'.format(model_num)].generation.no_repeat_ngram_size = 3

    # Fix seed for stochastic decoding
    if globals()['place_cfg_{}'.format(model_num)].common.seed is not None and not globals()['place_cfg_{}'.format(model_num)].generation.no_seed_provided:
        np.random.seed(globals()['place_cfg_{}'.format(model_num)].common.seed)
        utils.set_torch_seed(globals()['place_cfg_{}'.format(model_num)].common.seed)

    # Move models to GPU
    for globals()['place_model_{}'.format(model_num)] in globals()['place_models_{}'.format(model_num)]:
        globals()['place_model_{}'.format(model_num)].eval()
        if use_fp16:
            globals()['place_model_{}'.format(model_num)].half()
        if use_cuda and not globals()['place_cfg_{}'.format(model_num)].distributed_training.pipeline_model_parallel:
            globals()['place_model_{}'.format(model_num)].cuda()
        globals()['place_model_{}'.format(model_num)].prepare_for_inference_(globals()['place_cfg_{}'.format(model_num)])
        print("place model {} on the device".format(model_num))

    # Initialize generator
    globals()['place_generator_{}'.format(model_num)] = globals()['place_task_{}'.format(model_num)].build_generator(globals()['place_models_{}'.format(model_num)], globals()['place_cfg_{}'.format(model_num)].generation)

# Image transform
from torchvision import transforms
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((pick_cfg_135.task.patch_image_size, pick_cfg_135.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([pick_task_135.src_dict.bos()])
eos_item = torch.LongTensor([pick_task_135.src_dict.eos()])
pad_idx = pick_task_135.src_dict.pad()
def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = pick_task_135.tgt_dict.encode_line(
        line=pick_task_135.bpe.encode(text.lower()),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Construct input for refcoco task
patch_image_size = pick_cfg_135.task.patch_image_size
def construct_sample(image: Image, text: str):
    w, h = image.size
    w_resize_ratio = torch.tensor(patch_image_size / w).unsqueeze(0)
    h_resize_ratio = torch.tensor(patch_image_size / h).unsqueeze(0)
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        },
        "w_resize_ratios": w_resize_ratio,
        "h_resize_ratios": h_resize_ratio,
        "region_coords": torch.randn(1, 4)
    }
    return sample
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

# Function for scaling up bbox to original size
def scale_up(x1,y1,x2,y2):
    sx1 = x1
    sy1 = y1
    sx2 = x2
    sy2 = y2
    return int(sx1), int(sy1), int(sx2), int(sy2)

###############AAAAAAAAAAAAAAAAAA################
#################################################

pnp_image_root_path = './' # path to save image

HOST = '' # INSERT HOST (server) 
PORT = 9998

srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv_sock.bind((HOST, PORT))
srv_sock.listen()
cli_sock, addr = srv_sock.accept()
print(f'Connected by: {addr}')


while True:
    image_cnt = 0
    try:
        
        # Receive CV2 Image
        length = int(cli_sock.recv(65).decode('utf-8'), 2)
        print('sss')
        print(length)
        buf = b''
        while length:
            newbuf = cli_sock.recv(length)
            buf += newbuf
            length -= len(newbuf)
            print(length)
        print("image recieved from the robot!")
        
        data = np.frombuffer(base64.b64decode(buf), np.uint8)
        cv2_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # save received image
        save_image_path = pnp_image_root_path + '{}.png'.format(image_cnt)
        cv2.imwrite(save_image_path, cv2_img)
        image_cnt += 1
        
        #############################
        ###############vvvvvvvvvvvvvvvv################
        ### Visual Grounding HERE ###

        # Receive model time step input
        model_in = 0
        while model_in == 0:
            model_name = '135'
            if model_name in pick_model_path_dict.keys():
                print("evaluating on model train_{} and place_train_{}\n".format(model_name, model_name))
                model_in = 1
            else:
                print("wrong model name!")

        # Receive query number
        
        image = Image.open(save_image_path)
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        save_cropped_image_path = pnp_image_root_path + 'cropped_timestep{}_image_{}.png'.format(model_name, image_cnt)
        cv2.imwrite(save_cropped_image_path, image)

        image = Image.open(save_cropped_image_path)

        pick_instruction = input("Insert pick instruction ... ")
        place_instruction = input("Insert place instruction ... ")

        # Construct input sample & preprocess for GPU if cuda available
        rule_in = 0
        while rule_in == 0:
            rule_base = input("1 for rulebase placing, 0 for place model: ")
            # rule_base = 0
            if (int(rule_base) == 0) or (int(rule_base) == 1):
                rule_in = 1
        sample_pick = construct_sample(image, pick_instruction)
        sample_pick = utils.move_to_cuda(sample_pick) if use_cuda else sample_pick
        sample_pick = utils.apply_to_sample(apply_half, sample_pick) if use_fp16 else sample_pick

        place_rule = 'on the '
        if int(rule_base)==0:
            for rl in rel_list:
                if rl in place_instruction[:35]:
                    place_rule = rl
            sample_place = construct_sample(image, place_instruction)
        elif int(rule_base)==1:
            for rl in rel_list:
                if rl in place_instruction[:35]:
                    place_instruction = place_instruction.split(rl, 1)[-1]
                    place_rule = rl
                    print("rule base {} and place rule: {}".format(rule_base, place_rule))
            sample_place = construct_sample(image, place_instruction)
        sample_place = utils.move_to_cuda(sample_place) if use_cuda else sample_place
        sample_place = utils.apply_to_sample(apply_half, sample_place) if use_fp16 else sample_place

        # Run eval step for refcoco
        with torch.no_grad():
            result_pick, scores_pick = eval_step(globals()['pick_task_{}'.format(model_name)], globals()['pick_generator_{}'.format(model_name)], globals()['pick_models_{}'.format(model_name)], sample_pick)
            print("pick inferring with instruction...\n{}\n pick model...{}".format(pick_instruction, model_name))
            if int(rule_base) == 0:
                result_place, scores_place = eval_step(globals()['place_task_{}'.format(model_name)], globals()['place_generator_{}'.format(model_name)], globals()['place_models_{}'.format(model_name)], sample_place)
                print("place inferring with instruction...\n{}\n place model...{}".format(place_instruction, model_name))
            elif (int(rule_base) == 1) and (int(model_name) == 0):
                result_place, scores_place = eval_step(globals()['pick_task_{}'.format(model_name)], globals()['pick_generator_{}'.format(model_name)], globals()['pick_models_{}'.format(model_name)], sample_place)
                print("place inferring with instruction...\n{}\n pick model...{}".format(place_instruction, model_name))
            elif (int(rule_base) == 1) and (int(model_name) != 0):
                result_place, scores_place = eval_step(globals()['pick_task_{}'.format(model_name)], globals()['pick_generator_{}'.format(model_name)], globals()['pick_models_{}'.format(model_name)], sample_place)
                print("place inferring with instruction...\n{}\n pick model...{}".format(place_instruction, model_name))

        # Tmp bbox info
        xtl_pick, ytl_pick, xbr_pick, ybr_pick = result_pick[0]["box"][0], result_pick[0]["box"][1], result_pick[0]["box"][2], result_pick[0]["box"][3]
        xtl_place, ytl_place, xbr_place, ybr_place = result_place[0]["box"][0], result_place[0]["box"][1], result_place[0]["box"][2], result_place[0]["box"][3]
        scaled_xtl_pick, scaled_ytl_pick, scaled_xbr_pick, scaled_ybr_pick = scale_up(xtl_pick, ytl_pick, xbr_pick, ybr_pick)
        scaled_xtl_place, scaled_ytl_place, scaled_xbr_place, scaled_ybr_place = scale_up(xtl_place, ytl_place, xbr_place, ybr_place)
        place_coord_x = int((scaled_xtl_place + scaled_xbr_place)/2)
        place_coord_y = int((scaled_ytl_place + scaled_ybr_place)/2)
        x_ad = int(input("x adjustment ... "))
        y_ad = int(input("y adjustment ..."))
        ad_in = int(input("Do you want to adjust? ..."))
        if int(rule_base) == 1:
            if place_rule == rel_list[1]: #right
                place_coord_x += 100
                place_coord_y += 50
            elif place_rule == rel_list[2]: #left
                place_coord_x -= 100
                place_coord_y += 50
            elif place_rule == rel_list[3]: #front
                place_coord_y += 100            
            elif place_rule == rel_list[4]: #behind
                place_coord_y -= 100
        else:
            #place_coord_x += x_ad
            if place_rule == rel_list[1]: #right
                place_coord_y += 50
            elif place_rule == rel_list[2]: #left
                place_coord_y += 50
            elif place_rule == rel_list [3]: #front
                place_coord_y += 15
        if ad_in == 1:
            place_coord_x += x_ad
            place_coord_y += y_ad
        bbox_info = f'{scaled_xtl_pick};{scaled_ytl_pick};{scaled_xbr_pick};{scaled_ybr_pick};{place_coord_x};{place_coord_y}'
        print(f'Send {bbox_info}')
        print('\n')
        cli_sock.send(bbox_info.encode())

        ###############AAAAAAAAAAAAAAAAAA################
        #############################


    except KeyboardInterrupt:
        print('\n Server Ctrl-c')
        break
    #except IOError:
    #    print('\n IO Error')
    #    break
    except ValueError:
        print('\n Client Closed')
        break

cli_sock.close()
srv_sock.close()
