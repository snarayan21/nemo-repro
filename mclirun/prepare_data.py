import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 

    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

def download_and_extract_zip(url, dest_dir):
    zip_path = os.path.join(dest_dir, 'images.zip')
    download_file(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(dest_dir, 'images'))

    os.remove(zip_path)

def main():
    data_path = "dataset"
    os.makedirs(data_path, exist_ok=True)

    json_url = 'https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json'
    images_zip_url = 'https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip'

    json_dest_path = os.path.join(data_path, 'blip_laion_cc_sbu_558k.json')
    print(f"Downloading JSON metadata to {json_dest_path}...")
    download_file(json_url, json_dest_path)

    print(f"Downloading and extracting images to {data_path}...")
    download_and_extract_zip(images_zip_url, data_path)

if __name__ == "__main__":
    
    main()