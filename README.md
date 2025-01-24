## Setup

```bash
pixi install
```

## Training data

```
bold_conf|53859
epi_norm_rpt|1091370
skull_strip_report|478527
t1_norm_rpt|483651
tsnr_rpt|1086435
```

## Training

```bash=
env TORCH_LOGS="recompiles,graph_breaks" OMP_NUM_THREADS=10 python  \
    -m quality_control.autoencoder.cli \
    --datastore-database-uri="file:/scratch/imaging/quality-control/database-small.sqlite?mode=ro" \
    --data-module="TwoChannelDataModule" \
    --artifact-store-path="/scratch/imaging/quality-control/artifacts" \
    --batch-size=6 \
    --epoch-count=1

python  \
    -m quality_control.diffusion.cli \
    --datastore-database-uri="file:/scratch/imaging/quality-control/database-small.sqlite?mode=ro" \
    --data-module="TwoChannelDataModule" \
    --autoencoder-path="/scratch/imaging/quality-control/artifacts/dataset-database-small_model-autoencoder-l1-lpips_step-6000/autoencoder" \
    --optuna-database-uri="sqlite:///storage.sqlite" \
    --artifact-store-path="/scratch/imaging/quality-control/artifacts" \
    --batch-size=24 \
    --optuna-study-name="0"

python  \
    -m quality_control.diffusion.cli
    --datastore-database-uri="file:/data/cephfs-1/home/users/wallerl_c/work/quality-check/database-small.sqlite?mode=ro" \
    --data-module="DataModule1" \
    --optuna-database-uri="sqlite:///storage-3.sqlite" \
    --optuna-artifact-store-path="/data/cephfs-2/unmirrored/projects/walter-enigma-task-based-fmri/imaging/quality-check/artifacts" \
    --train-batch-size=24 --eval-batch-size=24 \
    --optuna-study-name="0"
```
