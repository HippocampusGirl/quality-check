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
env TORCH_LOGS="recompiles,graph_breaks" \
    TORCHINDUCTOR_CACHE_DIR=/scratch/imaging/quality-control/artifacts/cache \
    TORCHINDUCTOR_COMPILE_THREADS=1 \
    OMP_NUM_THREADS=10 python  \
    -m quality_control.autoencoder.cli \
    --datastore-database-uri="file:/scratch/imaging/quality-control/database-small.sqlite?mode=ro" \
    --data-module="TwoChannelDataModule" \
    --artifact-store-path="/scratch/imaging/quality-control/artifacts" \
    --batch-size=6 \
    --epoch-count=1

env OMP_NUM_THREADS=5 \
    TORCH_LOGS="recompiles,graph_breaks" \
    TORCHINDUCTOR_COMPILE_THREADS=1 \
    TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
    TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
    TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=1 \
    TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE=1 \
    TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=1 \
    TORCHINDUCTOR_REDIS_HOST=localhost \
    TORCHINDUCTOR_REDIS_PORT="${ports[1]}" \
    TRITON_CACHE_MANAGER=triton.runtime.cache:RemoteCacheManager \
    TRITON_REMOTE_CACHE_BACKEND=triton.runtime.cache:RedisRemoteCacheBackend \
    TRITON_REDIS_PORT="${ports[1]}" \
    python  \
    -m quality_control.diffusion.cli \
    --datastore-database-uri="file:/scratch/imaging/quality-control/database-small.sqlite?mode=ro" \
    --data-module="TwoChannelDataModule" \
    --autoencoder-path="/scratch/imaging/quality-control/dataset-database_model-autoencoder-l1-lpips_step-110000/autoencoder" \
    --optuna-database-uri="sqlite:///storage.sqlite" \
    --artifact-store-path="/scratch/imaging/quality-control/artifacts" \
    --batch-size=24 \
    --optuna-study-name="33"

python  \
    -m quality_control.diffusion.cli
    --datastore-database-uri="file:/data/cephfs-1/home/users/wallerl_c/work/quality-check/database-small.sqlite?mode=ro" \
    --data-module="DataModule1" \
    --optuna-database-uri="sqlite:///storage-3.sqlite" \
    --optuna-artifact-store-path="/data/cephfs-2/unmirrored/projects/walter-enigma-task-based-fmri/imaging/quality-check/artifacts" \
    --train-batch-size=24 --eval-batch-size=24 \
    --optuna-study-name="0"
```

```python
def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print('Tensor ', n[0], tensor.shape)
            except AttributeError as e:
                getBack(n[0])
```
