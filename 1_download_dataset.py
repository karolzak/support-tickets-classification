
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
        print('Done downloading file: '+file_path)
    else:
        print('File: ' + file_path + ' already exists.')


def download_dataset():
    print('Downloading Endava support tickets dataset...')
    folder_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        'datasets'
    )
    url = "https://privdatastorage.blob.core.windows.net/github/support-tickets-classification/datasets/all_tickets.csv?sp=r&st=2021-06-07T14:36:30Z&se=2022-12-30T23:36:30Z&spr=https&sv=2020-02-10&sr=b&sig=Za0%2Fgbe%2FanVblbcYsCdQS5zTS5%2B17QKESzlbEXPp2KE%3D"
    file_name = 'all_tickets.csv'
    download_file(url, folder_path, file_name)


if __name__ == "__main__":
    download_dataset()
