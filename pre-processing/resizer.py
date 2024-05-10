import torch 
from torchvision import transforms 
from PIL import Image 
import glob 
import os 
import numpy as np 

target_size = (256,256)

def resize_enum(target_dir, save_dir):
    files = [_ for _ in glob.glob(target_dir)]
    transform = transforms.Compose([
        transforms.Resize(target_size)
    ])
    for _ in range(len(files)):
        name = save_dir + files[_].split("\\")[-1]
        print(name)
        resized = transform(Image.open(files[_]))
        resized = np.array(resized)
        im = Image.fromarray(resized)
        im.save(name)
    
resize_enum(
    target_dir="one_more/train/Leprosy/*", 
    save_dir="wow_train/Leprosy/"
)
resize_enum(
    target_dir="one_more/train/non_leprosy/*", 
    save_dir="wow_train/Non_Leprosy/"
)
resize_enum(
    target_dir="one_more/test/Leprosy/*", 
    save_dir="wow_test/Leprosy/"
)
resize_enum(
    target_dir="one_more/test/non_leprosy/*", 
    save_dir="wow_test/Non_Leprosy/"
)

resize_enum(
    target_dir="one_more/valid/Leprosy/*", 
    save_dir="wow_valid/Leprosy/"
)
resize_enum(
    target_dir="one_more/valid/non_leprosy/*", 
    save_dir="wow_valid/Non_Leprosy/"
)