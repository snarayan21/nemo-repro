from omegaconf.omegaconf import OmegaConf
import torch.distributed

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import warnings
import datetime
import oci
import json
from ocifs import OCIFileSystem
import torch
warnings.filterwarnings("ignore", category=ResourceWarning)
import argparse
from PIL import Image
from streaming import StreamingDataset
from torchvision import transforms


def set_max_steps_from_streaming(cfg):
    if "oci://" not in cfg.model.data.data_path:
        raise ValueError("Data path should be an oci path")
    if cfg.trainer.max_epochs == -1:
        raise ValueError("Max epochs should not be -1 when using streaming data")


    oci_config = oci.config.from_file()
    oci_fs = OCIFileSystem(oci_config, region=cfg.model.data.region, oci_additional_kwargs={"retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY})

    with oci_fs.open(cfg.model.data.data_path + "/index.json", "r") as f:
        data = f.read()
        data = json.loads(data)

    total_data_samples = 0
    for s in data["shards"]:
        total_data_samples += s["samples"]

    cfg.model.global_batch_size = cfg.model.data.micro_batch_size * cfg.trainer.accumulate_grad_batches * cfg.trainer.devices * cfg.trainer.num_nodes // (cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size * cfg.model.context_parallel_size)
    cfg.trainer.accumulate_grad_batches = 1

    steps_per_epoch = total_data_samples // cfg.model.global_batch_size

    max_steps = steps_per_epoch
    
    cfg.trainer.max_steps = max_steps * cfg.trainer.max_epochs
    cfg.trainer.max_epochs = -1


class ImageDataset(StreamingDataset):
    def __init__(
        self,
        remote,
        local,
        shuffle,
        batch_size,
        region=None,
        replication=1,
    ):
        super(ImageDataset, self).__init__(
            remote=remote,
            local=local,
            shuffle=shuffle,
            batch_size=batch_size,
            replication=replication,
        )
        self.config = oci.config.from_file()
        print("Saaketh: oci config is: ", self.config)
        print("Saaketh: setting retry strategy.")
        self.oci_fs = OCIFileSystem(self.config, region=region, oci_additional_kwargs={"retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY})
        self.region = region
        self.transform = transforms.ToTensor()

    def handle_image(self, example):
        try:
            with self.oci_fs.open(
                example["image_url"],
                "rb",
                region=self.region,
                blocksize=4 * 1024 * 1024,
            ) as f:
                image = Image.open(f).convert("RGB")

        except Exception as e:
            print("image_dataloading error occured")
            print(e)
            import traceback
            traceback.print_exc()
            raise e

        image_tensor = self.transform(image)
        return image_tensor


    def __getitem__(self, i):
        try:
            if self.oci_fs is None:
                self.oci_fs = OCIFileSystem(self.config, region=self.region, oci_additional_kwargs={"retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY})

            data = super(ImageDataset, self).__getitem__(i)
            images = []

            url = data['content']['sources']["image_url"]
            example = {
                "image_url": url,
                "conversations": data['content']['sources']['conversations'],
            }

            image = self.handle_image(example)

            images.append(image)

            data_dict = {}
            data_dict["image"] = torch.cat(images)

            return data_dict

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            print("error caughted")
            data_dict = {}
            data_dict["image"] = torch.zeros(3, 100, 100, dtype=torch.float32)
            return data_dict


def main(dataset_path: str, region: str):

    torch.distributed.init_process_group()

    print("Saaketh: Initialized Process group!!")

    dataset = ImageDataset(
        remote=dataset_path,
        local=None,
        shuffle=False,
        batch_size=1,
        region=region,
    )

    curr_rank = torch.distributed.get_rank()

    for i, sample in enumerate(dataset):
        print(f'Saaketh: RANK {curr_rank}, Retrieved sample {i}')


if __name__ == '__main__':
    # Take in dataset path and region as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)
    args = parser.parse_args()
    main(args.dataset_path, args.region)