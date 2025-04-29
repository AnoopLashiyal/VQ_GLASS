# Data Reconstruction Attack: Latent Space Search (Rebranded GLASS)

## Overview
This repository implements a **Data Reconstruction Attack** using **latent space search** techniques.  
It is essentially a rebranding of **GLASS (GAN-based LAtent Space Search)**, where we aim to reconstruct private input data by optimizing in the generator's latent space.

The attack leverages a pretrained **Wasserstein GAN (WGAN) Generator** along with auxiliary models to achieve high-fidelity reconstructions from intermediate representations.

---

## Setup

### 1. Install Required Libraries
Make sure to install all necessary Python libraries. You can do so by running:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided, manually install libraries as indicated in the code.

### 2. Download Checkpoints
You need to download the pretrained checkpoints for:
- WGAN Generator
- Client/Server models
- Auxiliary models (if applicable)

👉 **Checkpoints can be downloaded from:**  
[Checkpoint Download Link](https://www.dropbox.com/scl/fi/bh79yfc3iv7fpe7gf8jz8/checkpoint.pth?rlkey=24lqtssalhkrpa5gt2e0q3llv&st=tlr9hldh&dl=1)  
[Discriminator Download Link](https://www.dropbox.com/scl/fi/5dh17y3szkxrz8gpz0vgr/discriminator_final.pth?rlkey=vuj3ew048pd20i82oxcicqypc&st=7l43jy9h&dl=1)  
[Generator Download Link](https://www.dropbox.com/scl/fi/nhp7nmsgk3xjkucmqv0l8/generator_final.pth?rlkey=c3wkqgocz8j4if27eq95jz92j&st=22ths9oj&dl=1)  
[Split Cifar Download Link](https://www.dropbox.com/scl/fi/45rtoid6uh723kc9f0u17/split_cifar_model.pth?rlkey=2zj03rb1y6vnu3bn3ujxmtjh6&st=bdlgx4fw&dl=1)  
[Inverse Net Download Link](https://www.dropbox.com/scl/fi/auwcypy7wi3fz1kk7w3qv/inverse_net.pth?rlkey=g2xsmjhj3h5s5nwc78dfia7i2&st=qfgup399&dl=1)  


---

## Usage

After setting up the environment and downloading the checkpoints:

Run the main script to start the reconstruction attack:
```bash
python main.py
```

You can also specify options manually:
```bash
python main.py --use_pretrained True --checkpoint_path /path/to/checkpoints
```

---

## Notes
- This project rebrands and extends the concepts introduced in GLASS, adapting them with improvements in latent optimization techniques.
- Make sure checkpoint paths are correctly specified either in the configuration files or as command-line arguments.
- Reconstruction quality depends on the fidelity of the generator and the structure of the intermediate representations.

---

## Citation
If you find this work useful, please consider citing GLASS and related research on latent space inversion attacks.

---

## License
This project is released under the MIT License. See `LICENSE` for more information.
