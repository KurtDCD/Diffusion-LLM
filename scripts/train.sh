
# This script is used to train diffusion policy

device=0

python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_guided \
    algo=diffusion_policy \
    exp_name=pretrain \
    variant_name=block_16 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.n_epochs=100 \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    train_dataloader.batch_size=256 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    rollout.enabled=false \
    device=cuda:${device} \
    seed=0

# Note1: change rollout.num_parallel_envs to 1 if libero vectorized env is not working as expected.
