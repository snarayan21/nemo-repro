rm -rf /tmp/e0e3d151499ba0716d5602efb3dcfece/
export NVIDIA_PYTORCH_VERSION=24.02

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 examples/multimodal/multimodal_llm/neva/neva_pretrain.py \
 ++cluster_type=BCP \
 trainer.precision=bf16 \
 trainer.num_nodes=1 \
 trainer.devices=2 \
 trainer.val_check_interval=100 \
 trainer.limit_val_batches=5 \
 trainer.log_every_n_steps=1 \
 trainer.max_epochs=1 \
 trainer.max_steps=-1 \
 trainer.check_val_every_n_epoch=1 \
 model.megatron_amp_O2=True \
 model.micro_batch_size=1 \
 model.global_batch_size=4 \
 model.tensor_model_parallel_size=2 \
 model.pipeline_model_parallel_size=1 \
 model.mcore_gpt=True \
 model.transformer_engine=True \
 model.data.data_path=oci://{oci-bucket}}/{streaming-dataset-path}/streaming \
 model.data.streaming=True \
 model.data.region=us-sanjose-1 \
 model.num_layers=32 \
 model.ffn_hidden_size=14336 \
 model.num_attention_heads=32 \
 model.normalization=rmsnorm \
 model.do_layer_norm_weight_decay=False \
 model.apply_query_key_layer_scaling=True \
 model.bias=False \
 model.activation=fast-swiglu \
 model.headscale=False \
 model.position_embedding_type=rope \
 model.rotary_percentage=1.0 \
 model.num_query_groups=8 \
 model.data.num_workers=4 \
 model.mm_cfg.llm.from_pretrained=checkpoints/llama-3_1-8b-instruct-nemo_v1.0/llama3_1_8b_instruct.nemo \
 model.mm_cfg.llm.model_type=llama_3 \
 model.data.conv_template=plain \
 model.mm_cfg.vision_encoder.from_pretrained='google/siglip-so400m-patch14-384' \
 model.mm_cfg.vision_encoder.from_hf=True \
 model.optim.name="fused_adam" \
 exp_manager.create_checkpoint_callback=True \
 exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
 exp_manager.create_wandb_logger=True \
 exp_manager.wandb_logger_kwargs.project="mcli-test"