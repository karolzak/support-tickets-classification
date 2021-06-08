from __future__ import print_function
import os
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


def download_file(file_url, folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        print('Downloading file from ' + file_url + '...')
        urlretrieve(file_url, file_path)
        print('Done downloading file: ' + file_path)
    else:
        print('File: ' + file_path + ' already exists.')


def download_models():
    print('Downloading models for web service...')
    file_list = ['category.model', 'impact.model', 'ticket_type.model']
    folder_path = os.path.dirname(os.path.abspath(__file__))
    url = "https://privdatastorage.blob.core.windows.net/github/support-tickets-classification/models/"
    sas = "?sp=r&st=2021-06-08T08:50:24Z&se=2022-12-31T17:50:24Z&spr=https&sv=2020-02-10&sr=c&sig=4z4Qa3PL2U%2B6TQiNgkU1MF%2BUiNCcvCbvgGXeLBW9LLg%3D"
    for file_name in file_list:
        download_file(url + file_name + sas, folder_path, file_name)


if __name__ == "__main__":
    download_models()
