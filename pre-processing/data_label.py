import os 
import glob 
import numpy as np
from PIL import Image

data = []
label = []

def prep_data(target_dir):
    for indx, filename in enumerate(os.listdir(target_dir)):
        file_fullpath = target_dir + "/" + filename 
        data.append(np.asarray(Image.open(file_fullpath)))
        if (target_dir.split("/")[-1]) == "Leprosy": 
            label.append(1)
            print("----> +")
        else: 
            label.append(0)

prep_data("new_train/Leprosy")
print("Non_Leprosy")
prep_data("new_train/Non_Leprosy")