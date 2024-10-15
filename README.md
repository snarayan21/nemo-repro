## Environment Setting (TBD)

## Model and Data Preparation

### Prepare pretrained model checkpoints

1) Download LLaMa-3.1 model from huggingface. We use 8B for debugging. <br>
Links: [LLama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct), [LLama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)

2) Convert the checkpoints into NeMo-compatible format by running the following code:

```bash
python scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
  --input_name_or_path {CHECKPOINT_DIR}/Meta-Llama-3.1-8B \
  --output_path /workspace/neva/checkpoints/llama-3.1-8b-instruct.nemo
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

## Launch pre-training

We run with tensor_parallel_size=4, pipeline_parallel_size=2.

```bash
mcli run -f mclirun/test.yaml
```
