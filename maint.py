import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import vgg19

class TemporalAnimeTransformer(nn.Module):
    def __init__(self, frame_size=(256, 256, 3), sequence_length=16):
        super(TemporalAnimeTransformer, self).__init__()
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.style_embedding_dim = 256
        self.content_embedding_dim = 512
        
        self.temporal_encoder = self.build_temporal_encoder()
        self.style_encoder = self.build_style_encoder()
        self.content_encoder = self.build_content_encoder()
        self.anime_generator = self.build_anime_generator()
        self.temporal_discriminator = self.build_temporal_discriminator()
        self.vgg = self.build_vgg()

        self.d_optimizer = torch.optim.Adam(self.temporal_discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(
            list(self.anime_generator.parameters()) + 
            list(self.temporal_encoder.parameters()) +
            list(self.content_encoder.parameters()) +
            list(self.style_encoder.parameters()), lr=1e-4, betas=(0.5, 0.999)
        )
        
    def build_temporal_encoder(self):
        return nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((self.sequence_length, self.frame_size[0]//16, self.frame_size[1]//16))
        )
    
    def build_style_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, self.style_embedding_dim)
        )
    
    def build_content_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, self.content_embedding_dim)
        )
    
    def build_anime_generator(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.content_embedding_dim + self.style_embedding_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def build_temporal_discriminator(self):
        return nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )
    
    def build_vgg(self):
        vgg = vgg19(pretrained=True)
        return nn.Sequential(*list(vgg.features)[:18])
    
    def adain(self, content, style):
        mean_content, std_content = content.mean([2, 3], keepdim=True), content.std([2, 3], keepdim=True)
        mean_style, std_style = style.mean([1], keepdim=True), style.std([1], keepdim=True)
        normalized = (content - mean_content) / (std_content + 1e-5)
        return normalized * std_style.unsqueeze(-1).unsqueeze(-1) + mean_style.unsqueeze(-1).unsqueeze(-1)
    
    def self_attention(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        return self.gamma * out + x
    
    def style_fusion_block(self, x, style):
        style_attention = self.fc(style).unsqueeze(-1).unsqueeze(-1)
        x = x * style_attention
        return x
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = F.relu(1.0 - real_output).mean()
        fake_loss = F.relu(1.0 + fake_output).mean()
        return real_loss + fake_loss
    
    def generator_loss(self, fake_output):
        return -fake_output.mean()
    
    def perceptual_loss(self, real_image, generated_image):
        real_features = self.vgg(real_image)
        generated_features = self.vgg(generated_image)
        return F.l1_loss(real_features, generated_features)
    
    def forward(self, real_sequence, anime_style_image):
        temporal_features = self.temporal_encoder(real_sequence)
        content_features = self.content_encoder(real_sequence[:, :, -1])
        style_features = self.style_encoder(anime_style_image)
        generated_frame = self.anime_generator(torch.cat([content_features, style_features], dim=1))
        return generated_frame
    
    def train_step(self, real_sequence, anime_style_image):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        
        temporal_features = self.temporal_encoder(real_sequence)
        content_features = self.content_encoder(real_sequence[:, :, -1])
        style_features = self.style_encoder(anime_style_image)
        
        generated_frame = self.anime_generator(torch.cat([content_features, style_features], dim=1))
        
        real_validity = self.temporal_discriminator(real_sequence)
        fake_sequence = torch.cat([real_sequence[:, :, :-1], generated_frame.unsqueeze(2)], dim=2)
        fake_validity = self.temporal_discriminator(fake_sequence)
        
        d_loss = self.discriminator_loss(real_validity, fake_validity)
        g_loss = self.generator_loss(fake_validity)
        content_loss = F.l1_loss(content_features, self.content_encoder(generated_frame))
        style_loss = F.l1_loss(style_features, self.style_encoder(generated_frame))
        temporal_consistency_loss = F.l1_loss(real_sequence[:, :, -1], generated_frame)
        perceptual_loss = self.perceptual_loss(real_sequence[:, :, -1], generated_frame)
        
        total_g_loss = g_loss + content_loss + style_loss + temporal_consistency_loss + perceptual_loss
        
        d_loss.backward(retain_graph=True)
        g_loss.backward()
        
        self.d_optimizer.step()
        self.g_optimizer.step()
        
        return d_loss.item(), total_g_loss.item()
    
    
