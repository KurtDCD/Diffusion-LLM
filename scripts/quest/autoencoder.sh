
# This script is used to train stage 0 i.e. the autoencoder of Quest

python train.py --config-name=train_autoencoder.yaml \
    task=libero_long \
    algo=quest \
    exp_name=final \
    variant_name=block_32_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    train_dataloader.batch_size=2048 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    device=cuda:2 \
    seed=0