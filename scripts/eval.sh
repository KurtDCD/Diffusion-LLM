
# python evaluate.py \
#     task=libero_long \
#     algo=quest \
#     exp_name=final \
#     variant_name=block_32_ds_4 \
#     stage=1 \
#     training.use_tqdm=false \
#     device=cuda:1 \
#     seed=0

algo=diffusion_policy
variant_name=block_16
device=0

python evaluate.py \
    task=metaworld_ml45_guided_test \
    algo=${algo} \
    exp_name=pretrain_test_guided \
    variant_name=${variant_name} \
    stage=1 \
    training.use_tqdm=true \
    make_unique_experiment_dir=true \
    checkpoint_path=/home/kurt/Diffusion-LLM/experiments/metaworld/ML45_GUIDED/diffusion_policy/pretrain/block_16/0/stage_1/multitask_model_epoch_0070.pth \
    update_cfg=true \
    device=cuda:${device} \
    rollout.rollouts_per_env=20 \
    rollout.n_video=10 \
    seed=0

# Note1: this will automatically load the latest checkpoint as per your exp_name, variant_name, algo, and stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.
