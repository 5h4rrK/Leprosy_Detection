import os 
import matplotlib.pyplot as plt
from PIL import Image 
import glob
import random as rd 
import numpy as np 

train_path_true = [ _ for _ in glob.glob("train/Leprosy/*")]
train_path_false = [_ for _ in glob.glob("train/non_leprosy/*")]

print("Training Data - Leprosy : ", train_path_true.__len__())
print("Training Data - Non_Leprosy : ", train_path_false.__len__())

def random_plot(lep, non_lep):
    m1 = rd.randint( 0, len(train_path_true) - 1)
    m2 = rd.randint( 0, len(train_path_false) - 1)
    sel1 =np.asarray(Image.open(lep[m1]).convert("RGB"))
    sel2 = np.asarray(Image.open(non_lep[m2]).convert("RGB"))
    # sel1 /= sel1.max()
    # sel2 /= sel2.max()
    # plt.figure(figsize=(20,6))
    # plt.subplot(1,2, 1)
    plt.imshow(sel1)
    # plt.subplot(1,2, 2)
    # plt.imshow(1,2,sel2 )
    plt.show()

# random_plot(train_path_true, train_path_false)
def rename_the_files(target_dir, target_name):
    files = [_ for _ in glob.glob(f"{target_dir}/*")]
    for _ in range(len(files)):
        try:
            os.system(f"mv {files[_]} {target_dir}/{target_name}_{_}.jpg")
        except: continue 

# rename_the_files("train/Leprosy", "Leprosy")
# rename_the_files("train/non_leprosy","Non_Leprosy")
# rename_the_files("valid/Leprosy","valid_Leprosy")
# rename_the_files("valid/non_leprosy","valid_Non_Leprosy") 
rename_the_files("test/Leprosy","test_Leprosy")
rename_the_files("test/non_leprosy","test_Non_Leprosy")