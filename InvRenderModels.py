import torch
from torch import nn
import numpy as np
import cupy as cp
from torch.autograd import Variable
import argparse
import random
import os
import models

class BrdfModel0(nn.Module):

    def __init__(self, device):
        super(BrdfModel0, self).__init__()
        modelsroot = 'models/check_cascade0_w320_h240'

        self.encoder = models.encoder0(cascadeLevel = 0).eval().to(device)
        self.encoder.load_state_dict(torch.load('{0}/encoder0_13.pth'.format(modelsroot)).state_dict())

        self.albedoDecoder = models.decoder0(mode=0).eval().to(device)
        self.albedoDecoder.load_state_dict(torch.load('{0}/albedo0_13.pth'.format(modelsroot)).state_dict())

        self.normalDecoder = models.decoder0(mode=1).eval().to(device)
        self.normalDecoder.load_state_dict(torch.load('{0}/normal0_13.pth'.format(modelsroot)).state_dict())

        self.roughDecoder = models.decoder0(mode=2).eval().to(device)
        self.roughDecoder.load_state_dict(torch.load('{0}/rough0_13.pth'.format(modelsroot)).state_dict())

        self.depthDecoder = models.decoder0(mode=4).eval().to(device)
        self.depthDecoder.load_state_dict(torch.load('{0}/depth0_13.pth'.format(modelsroot)).state_dict())

    def forward(self, img):
        with torch.no_grad():
            x1, x2, x3, x4, x5, x6 = self.encoder(img)

            albedoPred = 0.5 * (self.albedoDecoder(img, x1, x2, x3, x4, x5, x6) + 1)
            normalPred = self.normalDecoder(img, x1, x2, x3, x4, x5, x6)
            roughPred = self.roughDecoder(img, x1, x2, x3, x4, x5, x6 )
            depthPred = 0.5 * (self.depthDecoder(img, x1, x2, x3, x4, x5, x6) + 1)

        # Normalize Albedo and depth
        bn, ch, nrow, ncol = albedoPred.size()
        albedoPred = albedoPred.view(bn, -1)
        albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        albedoPred = albedoPred.view(bn, ch, nrow, ncol)

        bn, ch, nrow, ncol = depthPred.size()
        depthPred = depthPred.view(bn, -1)
        depthPred = depthPred / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        depthPred = depthPred.view(bn, ch, nrow, ncol)

        return albedoPred, normalPred, roughPred, depthPred


class BrdfModel1(nn.Module):

    def __init__(self, device):
        super(BrdfModel1, self).__init__()
        modelsroot = 'models/check_cascade1_w320_h240'

        self.encoder = models.encoder0(cascadeLevel = 1).eval().to(device)
        self.encoder.load_state_dict(torch.load('{0}/encoder1_6.pth'.format(modelsroot)).state_dict())

        self.albedoDecoder = models.decoder0(mode=0).eval().to(device)
        self.albedoDecoder.load_state_dict(torch.load('{0}/albedo1_6.pth'.format(modelsroot)).state_dict())

        self.normalDecoder = models.decoder0(mode=1).eval().to(device)
        self.normalDecoder.load_state_dict(torch.load('{0}/normal1_6.pth'.format(modelsroot)).state_dict())

        self.roughDecoder = models.decoder0(mode=2).eval().to(device)
        self.roughDecoder.load_state_dict(torch.load('{0}/rough1_6.pth'.format(modelsroot)).state_dict())

        self.depthDecoder = models.decoder0(mode=4).eval().to(device)
        self.depthDecoder.load_state_dict(torch.load('{0}/depth1_6.pth'.format(modelsroot)).state_dict())

    def forward(self, img, albedo, normal, rough, depth, diffuse, specular):

        input = torch.cat([img, albedo,
            0.5 * (normal+1), 0.5*(rough+1), depth,
            diffuse, specular], dim=1)

        with torch.no_grad():
            x1, x2, x3, x4, x5, x6 = self.encoder(input)

            albedoPred = 0.5 * (self.albedoDecoder(img, x1, x2, x3, x4, x5, x6) + 1)
            normalPred = self.normalDecoder(img, x1, x2, x3, x4, x5, x6)
            roughPred = self.roughDecoder(img, x1, x2, x3, x4, x5, x6 )
            depthPred = 0.5 * (self.depthDecoder(img, x1, x2, x3, x4, x5, x6) + 1)

        # Normalize Albedo and depth
        bn, ch, nrow, ncol = albedoPred.size()
        albedoPred = albedoPred.view(bn, -1)
        albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        albedoPred = albedoPred.view(bn, ch, nrow, ncol)

        bn, ch, nrow, ncol = depthPred.size()
        depthPred = depthPred.view(bn, -1)
        depthPred = depthPred / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        depthPred = depthPred.view(bn, ch, nrow, ncol)

        return albedoPred, normalPred, roughPred, depthPred
