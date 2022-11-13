import requests
from tqdm import tqdm

def download(url, fname):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)



url = "http://snap.stanford.edu/data/amazon/productGraph/image_features/categoryFiles/image_features_Sports_and_Outdoors.b"
download(url, "amazon-sports-outdoors.b")
url = "http://snap.stanford.edu/data/amazon/productGraph/image_features/categoryFiles/image_features_Clothing_Shoes_and_Jewelry.b"
download(url, "amazon-clothing-shoes-jewelry.b")
url = "http://snap.stanford.edu/data/amazon/productGraph/image_features/categoryFiles/image_features_Toys_and_Games.b"
download(url, "amazon-toys-games.b")
