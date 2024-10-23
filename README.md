## Docker Setting

Build the docker container with the following:

```bash
$ DOCKER_BUILDKIT=1 docker build -f Dockerfile.ci -t nemo:latest .
```

## Nvidia's docker build instruction (Recommend)

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
git clone https://github.com/laine-12labs/nemo-repro
cd nemo-repro
./reinstall.sh
```

After installing the repo, need to install apex, transformerengine, and megatron-core for LLM and Multi-modal project.

### From Nvidia's Docker

base image : nvcr.io/nvidia/pytorch:24.02-py3

#### Apex
```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c
pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"
```

#### TransformerEngine
```bash
git clone https://github.com/NVIDIA/TransformerEngine.git && \
cd TransformerEngine && \
git checkout bfe21c3d68b0a9951e5716fb520045db53419c5e && \
git submodule init && git submodule update && \
NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .
```

#### Megatron Core
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git && \
cd Megatron-LM && \
git checkout 02871b4df8c69fac687ab6676c4246e936ce92d0 && \
pip install . && \
cd megatron/core/datasets && \
make
```

### From Mosaic's Docker

base image : mosaicml/pytorch:2.4.0_cu124-python3.11-ubuntu20.04

NOTE : We couldn't install the packages from the commits above, so we found other way to install them.

#### Apex
```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
# install main branch
pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"
```

#### TransformerEngine
```bash
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
export NVTE_FRAMEWORK=pytorch
pip install .
```

If the Git repository has already been cloned, make sure to also clone the submodules
```bash
git submodule update --init --recursive
```

#### Megatron Core
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git && \
cd Megatron-LM && \
# install main branch
pip install .

# in our case, we couldn't execute below command, but it worked.
# not sure this causes the hangs.
cd megatron/core/datasets && \
make
```

Then assign this docker image in the yaml file.

## Launch pre-training

We run with tensor_parallel_size=4, pipeline_parallel_size=2. Run the following command.

```bash
mcli run -f mclirun/test.yaml
```

NOTE : If you build the docker file (not using nemo:latest), you should set nv_pytorch_tag like below.
```bash
export nv_pytorch_tag=24.02-py3
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

### Prepare the MDS dataset
After downloading `LAION/CC/SBU BLIP-Caption Concept-balanced 558K` by
```
python mclirun/prepare_data.py
```

You should upload all images to oci bucket.
```
oci os object sync --bucket-name {bucket_name} --src-dir "{dataset/LLaVA-Pretrain-LCS-558K/images}" --prefix "{bucket_path}/LLaVA-Pretrain-LCS-558K/images/" --include "*"
```
And make MDS.
```
python localrun/make_streaming_ds.py --json_path "{dataset/LLaVA-Pretrain-LCS-558K/blip_laion_cc_sbu_558k.json}" --remote "{bucket_path}/LLaVA-Pretrain-LCS-558K/streaming"
```


## Launch pretraining with MDS
### Local
Set your oci data path and region properly in localrun/test.sh<br>
`model.data.data_path=oci://{oci-bucket}}/{streaming-dataset-path}/streaming`<br>
`model.data.region={region}`
```
./localrun/test.sh
```

### MCLI
Set your oci data path region properly in mclirun/test_mds.sh<br>
`model.data.data_path=oci://{oci-bucket}}/{streaming-dataset-path}/streaming`<br>
`model.data.region={region}`<br>
Also, make the oci config. (/root/.oci)<br>
Launch the job.
```
mcli run -f mclirun/test_mds.yaml
```