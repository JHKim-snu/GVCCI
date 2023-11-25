#!/usr/bin/env bash

OMP_NUM_THREADS=4 python generate_instructions.py --vg_dataset ENV1_train --split_ind 0 --topn 1000 --each_image_query 10000;

