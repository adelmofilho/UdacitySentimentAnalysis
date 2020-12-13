import tarfile
from urllib.request import urlretrieve
from pathlib import Path
import os

def download_data(data_dir, data_url, filename):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.isfile(Path(data_dir).joinpath(filename)):
        download = urlretrieve(url=data_url, filename=Path(data_dir).joinpath(filename))
        tar = tarfile.open(name=download[0], mode='r|*')
        tar.extractall(data_dir)
        tar.close()
    else:
        print("data already downloaded")
