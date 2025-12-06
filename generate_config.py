import os
import shutil


def copy_file(input_pth, output_pth):
    shutil.copytree(
        src=input_pth,
        dst=output_pth,
        dirs_exist_ok=True
    )

def generate_kaggle_config():
    print('hello')
    home_dir = os.path.expanduser('~')
    kaggle_config_pth = os.path.join(home_dir, r".kaggle")
    is_file = os.path.isfile(kaggle_config_pth)
    #print(kaggle_config_pth, is_file)
    if not is_file:
        copy_file(r".kaggle", kaggle_config_pth)
        print('successfully created required Kaggle config file')
    else:
        print('Successfully located config file')
    
