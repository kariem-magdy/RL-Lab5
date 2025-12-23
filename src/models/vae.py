import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=32, img_channels=3, resize_dim=64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.resize_dim = resize_dim

        # Encoder
        # Output calculation: Floor((H + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
        # Here: Kernel=4, Stride=2, Padding=0 -> (H-4)/2 + 1 = H/2 - 1 roughly.
        # Ideally, we calculate the flattened size dynamically to support any resize_dim.
        
        self.enc_conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.enc_conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.enc_conv4 = nn.Conv2d(128, 256, 4, stride=2)
        
        # Calculate dynamic flatten size
        with torch.no_grad():
            self.flatten_dim = self._get_flatten_dim(img_channels, resize_dim)

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, self.flatten_dim)
        
        # We need to know the spatial dimensions before flattening to reshape in decode
        # Assuming 4 layers of stride 2 reduction on 64 -> 2x2.
        # On generic sizes, we reverse the calculation.
        self.last_h = int(resize_dim / (2**4)) # Approximation, typically valid for powers of 2
        if self.last_h < 1: self.last_h = 1
        # The channel depth at the last encoder layer is 256
        
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, 5, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.dec_conv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def _get_flatten_dim(self, c, h):
        x = torch.zeros(1, c, h, h)
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        x = self.enc_conv4(x)
        return int(torch.prod(torch.tensor(x.size())))

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        # Reshape to (Batch, Channel, H, W) matching the encoder's last conv output
        # Note: This exact shape logic assumes 64x64 -> 2x2x256. 
        # For full generalization, we should store the shape in _get_flatten_dim.
        # But for the 64x64 standard config, 256 ch, 2x2 spatial is correct.
        h = h.view(h.size(0), 256, 2, 2) 
        
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = F.relu(self.dec_conv3(h))
        return torch.sigmoid(self.dec_conv4(h)) 

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD