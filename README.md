# Diffusion Autoencoders
![demo_diffae](assets/demo.gif)

This is an unofficial implementation of [Diffusion Autoencoders](https://diff-ae.github.io/). You can find the official implementation [here](https://github.com/phizaz/diffae).

# :hammer_and_wrench: Setup
If you are using [poetry](https://github.com/python-poetry/poetry), you can install the required packages by running the following command:
```bash
poetry install
```

For more details on the required packages, refer to `pyproject.toml`.

# :rocket: Run
## Training Diff-AE
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

You can modify the settings before training. If you wish to create custom settings, modify the `diffae/cfg/{IMAGE_SIZE}_model.yml` file according to your GPU specs. :gear:

The training results will be saved in `output/{EXPN}`. If the --expn argument is not provided, the directory name will be generated based on the current time.

Note that Latent DDIM is not implemented as unconditional image synthesis by Diff-AE is not the focus of this repo.

## Evaluating Diff-AE
You can evaluate the trained model by calculating its MSE and [LPIPS](https://richzhang.github.io/PerceptualSimilarity/).

```bash
export PATH_TO_OUTPUT_DIR="output/hoge"
export MODEL_CKPT="last_ckpt.pth"

poetry run diffae_test \
    --output=$PATH_TO_OUTPUT_DIR \
    --model_ckpt=$MODEL_CKPT
```

## Training the Classifier
You can also train a classifier for attribute manipulation.

```bash
export PATH_TO_OUTPUT_DIR="output/hoge"
export MODEL_CKPT="last_ckpt.pth"

poetry run clf_train \
    --output=$PATH_TO_OUTPUT_DIR \
    --model_ckpt=$MODEL_CKPT
```

## Evaluating the Classifier
The classifier can be evaluated by calculating its accuracy and AUROC.

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
You can check out the minimal working examples in the notebooks found [here](demo).
- [autoencode.ipynb](demo/autoencode.ipynb)
- [manipulation.ipynb](demo/manipulation.ipynb)
- [interpolation.ipynb](demo/interpolation.ipynb)

---

If you find this repository helpful, please consider giving a star :star:!
