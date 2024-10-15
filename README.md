## Docker Setting

Build the docker container with the following:

```bash
$ DOCKER_BUILDKIT=1 docker build -f Dockerfile.ci -t nemo:latest .
```

Then assign this docker image in the yaml file.

## Launch pre-training

We run with tensor_parallel_size=4, pipeline_parallel_size=2. Run the following command.

```bash
mcli run -f mclirun/test.yaml
```

## Model and Data Preparation (optional)

Model checkpoint and dataset are prepared on the fly within the yaml file. For further information, refer to the contents below.

### Prepare pretrained model checkpoints

Download `.nemo` format checkpoint for LLAMA-3.1 at [this link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/llama-3_1-8b-instruct-nemo). Alternatively, you can use the following command:<br>

```bash
$ ngc registry model download-version "nvidia/nemo/llama-3_1-8b-instruct-nemo:1.0"
```

### Prepare the dataset

Download `LAION/CC/SBU BLIP-Caption Concept-balanced 558K` dataset from [Official LLaVA repository](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md).<br>
The directory structure would be like:<br>
```
DATA_DIR
  ㄴ LLaVA-Pretrain-LCS-558K
    ㄴ blip_laion_cc_sbu_558k.json
    ㄴ images/
```