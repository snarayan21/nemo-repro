# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf.omegaconf import OmegaConf

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
warnings.filterwarnings("ignore", category=ResourceWarning)


def set_max_steps_from_streaming(cfg):
    if "oci://" not in cfg.model.data.data_path:
        raise ValueError("Data path should be an oci path")
    if cfg.trainer.max_epochs == -1:
        raise ValueError("Max epochs should not be -1 when using streaming data")


    oci_config = oci.config.from_file()
    oci_fs = OCIFileSystem(oci_config, region=cfg.model.data.region)

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


@hydra_runner(config_path="conf", config_name="neva_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    cur_datetime = datetime.datetime.now().strftime("%y%m%d-%H%M")
    cfg.exp_manager.name = cur_datetime

    if cfg.model.data.streaming:
        set_max_steps_from_streaming(cfg)

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model = MegatronNevaModel(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()