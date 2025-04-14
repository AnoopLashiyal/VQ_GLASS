import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from omegaconf import OmegaConf
import urllib.request
from taming.models.vqgan import VQModel
import requests

# ========== 1. Load local VQ-GAN model ==========
def load_local_vqgan(config_path="model.yaml", state_dict_path="last.ckpt", device="mps" if torch.cuda.is_available() else "cpu"):
    print(f"Loading VQGAN from {state_dict_path}")
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    model.eval()
    model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model

# ========== 2. Dataset loader ==========
def get_dataset_loader(image_dir, batch_size=4, image_size=256):
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    dataset = datasets.ImageFolder(root=image_dir, transform=tf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ========== 3. Client model ==========
class ClientModel(torch.nn.Module):
    def __init__(self, vqgan):
        super().__init__()
        self.encoder = vqgan.encoder
        self.quant_conv = vqgan.quant_conv
        self.quantize = vqgan.quantize

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.quant_conv(z_e)
        z_q, _, _ = self.quantize(z_e)
        return z_q

# ========== 4. Server decoder ==========
class ServerDecoder(torch.nn.Module):
    def __init__(self, vqgan):
        super().__init__()
        self.decode = vqgan.decode

    def forward(self, z_q):
        return self.decode(z_q)

# ========== 5. Save image pairs ==========
def save_images(original, reconstructed, save_dir="results", start_index=0):
    os.makedirs(save_dir, exist_ok=True)
    original = (original + 1) / 2
    reconstructed = (reconstructed + 1) / 2
    bs = original.shape[0]

    for i in range(bs):
        idx = start_index + i
        save_image(original[i], os.path.join(save_dir, f"original_{idx}.png"))
        save_image(reconstructed[i], os.path.join(save_dir, f"reconstructed_{idx}.png"))

# ========== 6. Main ==========
if __name__ == "__main__":
    # Dataset directory
    dataset_path = "/Users/anooplashiyal/datasets/flowers"  # <-- your folder should contain the subfolder 'jpg'
    subfolder = "jpg"

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = "model.yaml"
    ckpt_path = "last.ckpt"

    # Load VQ-GAN and create client/server
    vqgan = load_local_vqgan(config_path, ckpt_path).to(device)
    client = ClientModel(vqgan).to(device)
    server = ServerDecoder(vqgan).to(device)

    # Load dataset
    dataloader = get_dataset_loader(os.path.join(dataset_path, subfolder), batch_size=4)

    # Run reconstruction
    with torch.no_grad():
        image_index = 0
        for batch, _ in dataloader:
            batch = batch.to(device)
            z_q = client(batch)
            x_hat = server(z_q)
            save_images(batch, x_hat, save_dir="results", start_index=image_index)
            image_index += batch.size(0)