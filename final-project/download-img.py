# download image

import wget
import os
import pandas as pd #for dataframe operations

# define 
dataset_dir = "dress-10k.csv"
img_dir = "img"

#Import the dataset
df = pd.read_csv(dataset_dir)
df.head(10)

# make directory
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
    
for url in df['image_url'] :
    local_file = wget.download(url, img_dir)
    print(local_file)