import os
import shutil
import generate_config

data_files = ['Answers.csv', 'Questions.csv', 'Tags.csv']

def data_installed(download_pth = 'data'):
    for file in data_files:
        pth = os.path.join(download_pth, file)
        is_file = os.path.isfile(pth)
        if not is_file:
            return False
    return True

def connect_api():
    print('connecting to API...')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def install_data(dataset ='stackoverflow/stacksample', download_pth = 'data'):
    if not data_installed(download_pth):
        print('installing dataset...')
        api = connect_api()
        print(f'saving dataset to {download_pth}...')
        api.dataset_download_files(dataset, path=download_pth, unzip=True)
    else:
        print('dataset already installed')
