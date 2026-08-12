#!/usr/bin/env python3
from glob import glob
import re
import os 

write_file = open("./data/paddleline/arabic_train.txt", "w")
for file in glob("./data/synthetic-ids/train/line/*.png"):
    file_name = re.match(r"./data/synthetic-ids/train/line/(.*)\.png", file).group(1)
    with open(f"./data/synthetic-ids/train/line/{file_name}.gt.txt", 'r') as f:
        write_file.write(f"images/{file_name}.png" + "\t" + f.read().strip() + "\n")





write_file = open("./data/paddleline/arabic_val.txt", "w")
for file in glob("./data/synthetic-ids/val/line/*.png"):
    file_name = re.match(r"./data/synthetic-ids/val/line/(.*)\.png", file).group(1)
    with open(f"./data/synthetic-ids/val/line/{file_name}.gt.txt", 'r') as f:
        write_file.write(f"images/{file_name}_val.png" + "\t" + f.read().strip() + "\n")
