import os

def get_list(path, list_type='img'):
    my_list = []
    if list_type == 'img':
        valid_list = ['.jpg', '.png']
    # get all image name
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if not (ext.lower() in valid_list):
            continue
        my_list.append(f)
    return my_list

def check_dir(path):
    # make directory
    if not os.path.exists(path):
        os.makedirs(path)