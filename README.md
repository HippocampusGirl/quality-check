## Setup

```bash
micromamba create --name "machine-learning" \
    "python=3.12" \
    "jupyter" \
    "tqdm" \
    "numpy" "scipy" "seaborn" "pandas" \
    "pillow" "cairosvg" \
    "diffusers" "safetensors" \
    "pytorch" "torchio" "torchmetrics" "torchvision" \
    "jaxlib=*=cuda120*" "jax" "chex" "flax" "jaxtyping" \
    "numpyro"
```
