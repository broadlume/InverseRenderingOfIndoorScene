import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import models

class BrdfModel0(nn.Module):

    def __init__(self, imgWidth, imgHeight, device):
        super(BrdfModel0, self).__init__()

        self.imgWidth = imgWidth
        self.imgHeight = imgHeight

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

    def __init__(self, imgWidth, imgHeight, device):
        super(BrdfModel1, self).__init__()

        self.imgWidth = imgWidth
        self.imgHeight = imgHeight

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


class LightModel(nn.Module):

    def __init__(self, level, imgWidth, imgHeight, envWidth, envHeight, envRenderWidth, envRenderHeight, device):
        super(LightModel, self).__init__()

        self.level = level
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
        self.envWidth = envWidth
        self.envHeight = envHeight
        self.envRenderWidth = envRenderWidth
        self.envRenderHeight = envRenderHeight

        isCuda = "cuda" in str(device)
        modelsroot = 'models/check_cascadeLight{0}_sg12_offset1'.format(level)
        SGNum = 12

        self.lightEncoder = models.encoderLight(cascadeLevel = level, SGNum = SGNum).eval().to(device)
        self.lightEncoder.load_state_dict(torch.load('{0}/lightEncoder{1}_9.pth'.format(modelsroot, level)).state_dict())

        self.axisDecoder = models.decoderLight(mode=0, SGNum = SGNum ).eval().to(device)
        self.axisDecoder.load_state_dict(torch.load('{0}/axisDecoder{1}_9.pth'.format(modelsroot, level) ).state_dict())

        self.lambDecoder = models.decoderLight(mode=1, SGNum = SGNum ).eval().to(device)
        self.lambDecoder.load_state_dict(torch.load('{0}/lambDecoder{1}_9.pth'.format(modelsroot, level) ).state_dict())

        self.weightDecoder = models.decoderLight(mode=2, SGNum = SGNum ).eval().to(device)
        self.weightDecoder.load_state_dict(torch.load('{0}/weightDecoder{1}_9.pth'.format(modelsroot, level) ).state_dict())    


    def forward(self, img, imgSmall, albedo, normal, rough, depth, envmaps = None):

        # Interpolation
        imBatchLarge = F.interpolate(img, [imgSmall.size(2) *
            4, imgSmall.size(3) * 4], mode='bilinear')
        albedoPredLarge = F.interpolate(albedo, [imgSmall.size(2)*
            4, imgSmall.size(3) * 4], mode='bilinear')
        normalPredLarge = F.interpolate(normal, [imgSmall.size(2) *
            4, imgSmall.size(3) * 4], mode='bilinear')
        roughPredLarge = F.interpolate(rough, [imgSmall.size(2) *
            4, imgSmall.size(3) * 4], mode='bilinear')
        depthPredLarge = F.interpolate(depth, [imgSmall.size(2) *
            4, imgSmall.size(3) * 4], mode='bilinear')

        input = torch.cat([imBatchLarge, albedoPredLarge,
                    0.5*(normalPredLarge+1), 0.5*(roughPredLarge+1), depthPredLarge ], dim=1 )

        with torch.no_grad():
            x1, x2, x3, x4, x5, x6 = self.lightEncoder(input, envmaps if self.level==1 else None)

            # Prediction
            axisPred = self.axisDecoder(x1, x2, x3, x4, x5, x6, imgSmall )
            lambPred = self.lambDecoder(x1, x2, x3, x4, x5, x6, imgSmall )
            weightPred = self.weightDecoder(x1, x2, x3, x4, x5, x6, imgSmall )

        bn, SGNum, _, envRow, envCol = axisPred.size()
        envmapsPred = torch.cat([axisPred.view(bn, SGNum*3, envRow, envCol ), lambPred, weightPred], dim=1)

        self.renderLayer = models.renderingLayer(isCuda = True,
                imWidth = imgSmall.shape[-1], imHeight = imgSmall.shape[-2], fov = 56,
                envWidth = self.envRenderWidth, envHeight = self.envRenderHeight)

        self.output2env = models.output2env(isCuda = True,
                envWidth = self.envRenderWidth, envHeight = self.envRenderHeight, SGNum = SGNum )

        envmapsPredImage, axisPred, lambPred, weightPred = self.output2env.output2env(axisPred, lambPred, weightPred )

        diffusePred, specularPred = self.renderLayer.forwardEnv(albedo, normal,
                rough, envmapsPredImage)
        
        return diffusePred, specularPred, envmapsPred
