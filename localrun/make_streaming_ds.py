from tqdm import tqdm
import json
from streaming import JSONWriter, MDSWriter
import os
import argparse

def main(args):
    JSON_PATH = args.json_path
    REMOTE = args.remote

    columns = {
        "content": "json",
    }

    samples = []

    with open(JSON_PATH, 'r') as f:
        file_list = json.load(f)

    for qa_pair in tqdm(file_list):
        id = qa_pair['id']
        image_path = qa_pair['image']
        conversations = qa_pair['conversations']

        image_path = os.path.join(REMOTE, 'images', image_path)

        samples.append({
            "content": {
                "sources": {
                    "id": id,
                    "image_url": image_path,
                    "conversations": conversations
                }
            }
        })

    with MDSWriter(out=os.path.join(REMOTE, 'streaming'), columns=columns) as out:
        for sample in tqdm(samples):
            out.write(sample)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file')
    parser.add_argument('--remote', type=str, required=True, help='Remote path for MDSWriter')
    args = parser.parse_args()

    main(args)