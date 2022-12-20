# Diffusion Autoencoders
![demo_diffae](assets/demo.gif)

Unofficial implementation of [Diffusion Autoencoders](https://diff-ae.github.io/).<br>
The official implementation is [here](https://github.com/phizaz/diffae).

# :hammer_and_wrench: Setup
If you are using [poetry](https://github.com/python-poetry/poetry), run the following command.
```bash
poetry install
```

See `pyproject.toml` for more details on required packages.

# :rocket: Run
## Diff-AE training
You can train Diff-AE on [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or [CelebA-HQ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html) datasets.<br>
**There is no need to download the datasets manually**; they are automatically downloaded when you run.

```bash
export DATA_NAME="celebahq"
export IMAGE_SIZE=128
export EXPN="hoge"

poetry run diffae_train \
    --data_name=$DATA_NAME \
    --image_size=$IMAGE_SIZE \
    --expn=$EXPN
```

You can change the settings before training. If you want to make your own settings, modify `diffae/cfg/{IMAGE_SIZE}_model.yml` according to your GPU specs. :gear:

The training results are saved under `output/{EXPN}`. If the --expn argument is not given, the directory name is assigned according to the time of day.

Latent DDIM is not implemented as I am not interested in unconditional image synthesis by Diff-AE.

## Diff-AE evaluation
You can evaluate the model by calculating MSE and [LPIPS](https://richzhang.github.io/PerceptualSimilarity/).

```bash
export PATH_TO_OUTPUT_DIR="output/hoge"
export MODEL_CKPT="last_ckpt.pth"

poetry run diffae_test \
    --output=$PATH_TO_OUTPUT_DIR \
    --model_ckpt=$MODEL_CKPT
```

## Classifier Training
You can train classifier for attribute manipulation.

```bash
export PATH_TO_OUTPUT_DIR="output/hoge"
export MODEL_CKPT="last_ckpt.pth"

poetry run clf_train \
    --output=$PATH_TO_OUTPUT_DIR \
    --model_ckpt=$MODEL_CKPT
```

## Classifier evaluation
You can evaluate the classifier by calculating accuracy and AUROC.

```bash
export PATH_TO_OUTPUT_DIR="output/hoge"
export MODEL_CKPT="last_ckpt.pth"
export CLF_CKPT="clf_last_ckpt.pth"

poetry run clf_test \
    --output=$PATH_TO_OUTPUT_DIR \
    --model_ckpt=$MODEL_CKPT \
    --clf_ckpt=$CLF_CKPT
```

# :books: Example Notebooks
You can view the minimal working notebooks [here](demo).
- [autoencode.ipynb](demo/autoencode.ipynb)
- [manipulation.ipynb](demo/manipulation.ipynb)
- [interpolation.ipynb](demo/interpolation.ipynb)

---

If this repo is helpful, please give me a star :star:
