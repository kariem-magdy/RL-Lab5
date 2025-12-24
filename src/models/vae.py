import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=32, img_channels=3, resize_dim=64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.resize_dim = resize_dim

        self.enc_conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.enc_conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.enc_conv4 = nn.Conv2d(128, 256, 4, stride=2)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, img_channels, resize_dim, resize_dim)
            x = self.enc_conv1(dummy_input)
            x = self.enc_conv2(x)
            x = self.enc_conv3(x)
            x = self.enc_conv4(x)
            
            self.bottleneck_shape = x.shape[1:] 
            self.flatten_dim = int(torch.prod(torch.tensor(self.bottleneck_shape)))

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, self.flatten_dim)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, 5, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.dec_conv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

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
        h = h.view(h.size(0), *self.bottleneck_shape)
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