import torch
import numpy as np
import cupy;
from torch.autograd import Variable
import argparse
import random
import os
import models
import utils
import glob
import os.path as osp
import cv2
import BilateralLayer as bs
import torch.nn.functional as F
import scipy.io as io
import utils

import time

starttick = time.time()

parser = argparse.ArgumentParser()
# The locationi of testing set
parser.add_argument('--dataRoot', help='path to real images')
parser.add_argument('--imList', help='path to image list')

parser.add_argument('--experiment0', default=None, help='the path to the model of first cascade' )
parser.add_argument('--experimentLight0', default=None, help='the path to the model of first cascade' )
parser.add_argument('--experiment1', default=None, help='the path to the model of second cascade' )
parser.add_argument('--experimentLight1', default=None, help='the path to the model of second cascade')

parser.add_argument('--testRoot', help='the path to save the testing errors' )

# The basic testing setting
parser.add_argument('--nepoch0', type=int, default=14, help='the number of epoch for testing')
parser.add_argument('--nepochLight0', type=int, default=10, help='the number of epoch for testing')

parser.add_argument('--nepoch1', type=int, default=7, help='the number of epoch for testing')
parser.add_argument('--nepochLight1', type=int, default=10, help='the number of epoch for testing')

parser.add_argument('--imHeight0', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth0', type=int, default=320, help='the height / width of the input image to network' )
parser.add_argument('--imHeight1', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth1', type=int, default=320, help='the height / width of the input image to network' )

parser.add_argument('--envRow', type=int, default=120, help='the height /width of the envmap predictions')
parser.add_argument('--envCol', type=int, default=160, help='the height /width of the envmap predictions')
parser.add_argument('--envHeight', type=int, default=8, help='the height /width of the envmap predictions')
parser.add_argument('--envWidth', type=int, default=16, help='the height /width of the envmap predictions')

parser.add_argument('--SGNum', type=int, default=12, help='the number of spherical Gaussian lobes')
parser.add_argument('--offset', type=float, default=1, help='the offset when train the lighting network')

parser.add_argument('--cuda', action = 'store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for testing network')

parser.add_argument('--level', type=int, default=2, help='the cascade level')
parser.add_argument('--isLight', action='store_true', help='whether to predict lightig')
parser.add_argument('--isBS', action='store_true', help='whether to use bilateral solver')

# Image Picking
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

if opt.experiment0 is None:
    opt.experiment0 = 'check_cascade0_w%d_h%d' % (opt.imWidth0, opt.imHeight0 )

if opt.experiment1 is None:
    opt.experiment1 = 'check_cascade1_w%d_h%d' % (opt.imWidth1, opt.imHeight1 )

if opt.experimentLight0 is None:
    opt.experimentLight0 = 'check_cascadeLight0_sg%d_offset%.1f' % \
            (opt.SGNum, opt.offset )

if opt.experimentLight1 is None:
    opt.experimentLight1 = 'check_cascadeLight1_sg%d_offset%.1f' % \
            (opt.SGNum, opt.offset )

experiments = [opt.experiment0, opt.experiment1 ]
experimentsLight = [opt.experimentLight0, opt.experimentLight1 ]
nepochs = [opt.nepoch0, opt.nepoch1 ]
nepochsLight = [opt.nepochLight0, opt.nepochLight1 ]

imHeight = opt.imHeight0
imWidth = opt.imWidth0

os.system('mkdir {0}'.format(opt.testRoot ) )
os.system('cp *.py %s' % opt.testRoot )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

opt.batchSize = 1
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


encoders = []
albedoDecoders = []
normalDecoders = []
roughDecoders = []
depthDecoders = []

lightEncoders= []
axisDecoders = []
lambDecoders = []
weightDecoders = []

startloadtick = time.time()

imBatchSmall = Variable(torch.FloatTensor(opt.batchSize, 3, opt.envRow, opt.envCol ) )
for n in range(0, opt.level ):
    # BRDF Predictioins
    encoders.append(models.encoder0(cascadeLevel = n).eval()  )
    albedoDecoders.append(models.decoder0(mode=0).eval() )
    normalDecoders.append(models.decoder0(mode=1).eval() )
    roughDecoders.append(models.decoder0(mode=2).eval() )
    depthDecoders.append(models.decoder0(mode=4).eval() )

    # Load weight
    encoders[n].load_state_dict(
            torch.load('{0}/encoder{1}_{2}.pth'.format(experiments[n], n, nepochs[n]-1) ).state_dict() )
    albedoDecoders[n].load_state_dict(
            torch.load('{0}/albedo{1}_{2}.pth'.format(experiments[n], n, nepochs[n]-1) ).state_dict() )
    normalDecoders[n].load_state_dict(
            torch.load('{0}/normal{1}_{2}.pth'.format(experiments[n], n, nepochs[n]-1) ).state_dict() )
    roughDecoders[n].load_state_dict(
            torch.load('{0}/rough{1}_{2}.pth'.format(experiments[n], n, nepochs[n]-1) ).state_dict() )
    depthDecoders[n].load_state_dict(
            torch.load('{0}/depth{1}_{2}.pth'.format(experiments[n], n, nepochs[n]-1) ).state_dict() )

    for param in encoders[n].parameters():
        param.requires_grad = False
    for param in albedoDecoders[n].parameters():
        param.requires_grad = False
    for param in normalDecoders[n].parameters():
        param.requires_grad = False
    for param in roughDecoders[n].parameters():
        param.requires_grad = False
    for param in depthDecoders[n].parameters():
        param.requires_grad = False

    if opt.isLight or (opt.level == 2 and n == 0):
        # Light network
        lightEncoders.append(models.encoderLight(cascadeLevel = n, SGNum = opt.SGNum).eval() )
        axisDecoders.append(models.decoderLight(mode=0, SGNum = opt.SGNum ).eval() )
        lambDecoders.append(models.decoderLight(mode=1, SGNum = opt.SGNum ).eval() )
        weightDecoders.append(models.decoderLight(mode=2, SGNum = opt.SGNum ).eval() )

        lightEncoders[n].load_state_dict(
                torch.load('{0}/lightEncoder{1}_{2}.pth'.format(experimentsLight[n], n, nepochsLight[n]-1) ).state_dict() )
        axisDecoders[n].load_state_dict(
                torch.load('{0}/axisDecoder{1}_{2}.pth'.format(experimentsLight[n], n, nepochsLight[n]-1) ).state_dict() )
        lambDecoders[n].load_state_dict(
                torch.load('{0}/lambDecoder{1}_{2}.pth'.format(experimentsLight[n], n, nepochsLight[n]-1) ).state_dict() )
        weightDecoders[n].load_state_dict(
                torch.load('{0}/weightDecoder{1}_{2}.pth'.format(experimentsLight[n], n, nepochsLight[n]-1) ).state_dict() )

        for param in lightEncoders[n].parameters():
            param.requires_grad = False
        for param in axisDecoders[n].parameters():
            param.requires_grad = False
        for param in lambDecoders[n].parameters():
            param.requires_grad = False
        for param in weightDecoders[n].parameters():
            param.requires_grad = False

#########################################

print("Loaded models {}".format(time.time()-startloadtick))
togpustarttick = time.time()

##############  ######################
# Send things into GPU
if opt.cuda:
    for n in range(0, opt.level ):
        encoders[n] = encoders[n].cuda(opt.gpuId )
        albedoDecoders[n] = albedoDecoders[n].cuda(opt.gpuId )
        normalDecoders[n] = normalDecoders[n].cuda(opt.gpuId )
        roughDecoders[n] = roughDecoders[n].cuda(opt.gpuId )
        depthDecoders[n] = depthDecoders[n].cuda(opt.gpuId )

        if opt.isLight or (n == 0 and opt.level == 2):
            lightEncoders[n] = lightEncoders[n].cuda(opt.gpuId )
            axisDecoders[n] = axisDecoders[n].cuda(opt.gpuId )
            lambDecoders[n] = lambDecoders[n].cuda(opt.gpuId )
            weightDecoders[n] = weightDecoders[n].cuda(opt.gpuId )
####################################

print("Sent models to gpu {}".format(time.time()-togpustarttick))

####################################
outfilename = opt.testRoot + '/results'
for n in range(0, opt.level ):
    outfilename = outfilename + '_brdf%d' % nepochs[n]
    if opt.isLight:
        outfilename += '_light%d' % nepochsLight[n]
os.system('mkdir -p {0}'.format(outfilename ) )

with open(opt.imList, 'r') as imIdIn:
    imIds = imIdIn.readlines()
imList = [osp.join(opt.dataRoot,x.strip() ) for x in imIds ]
imList = sorted(imList )

j = 0
for imName in imList:
    procimgstarttick = time.time()
    j += 1
    print('%d/%d: %s' % (j, len(imList), imName) )

    imBatches = []
    shadingNames = []

    imId = imName.split('/')[-1]
    print(imId )

    for n in range(0, opt.level ):
        shadingNames.append(osp.join(outfilename, imId.replace('.png', '_shading%d.png' % n) ) )

    # Load the image from cpu to gpu
    assert(osp.isfile(imName ) )
    im_cpu = cv2.imread(imName )[:, :, ::-1]
    nh, nw = im_cpu.shape[0], im_cpu.shape[1]

    # Resize Input Images
    resizedWidth = 0
    resizedHeight = 0

    if nh < nw:
        resizedWidth = imWidth
        resizedHeight = int(float(imWidth ) / float(nw) * nh )
    else:
        resizedHeight = imHeight
        resizedWidth = int(float(imHeight ) / float(nh) * nw )

    if nh < resizedHeight:
        im = cv2.resize(im_cpu, (resizedWidth, resizedHeight), interpolation = cv2.INTER_AREA )
    else:
        im = cv2.resize(im_cpu, (resizedWidth, resizedHeight), interpolation = cv2.INTER_LINEAR )

    im = (np.transpose(im, [2, 0, 1] ).astype(np.float32 ) / 255.0 )[np.newaxis, :, :, :]
    im = im / im.max()
    imBatches.append( Variable(torch.from_numpy(im**(2.2) ) ).cuda() )

    nh, nw = resizedHeight, resizedWidth

    newEnvWidth, newEnvHeight, fov = 0, 0, 0
    if resizedHeight < resizedWidth:
        fov = 57
        newEnvWidth = opt.envCol
        newEnvHeight = int(float(opt.envCol ) / float(resizedWidth) * resizedHeight )
    else:
        fov = 42.75
        newEnvHeight = opt.envRow
        newEnvWidth = int(float(opt.envRow ) / float(resizedHeight) * resizedWidth )

    if resizedHeight < newEnvHeight:
        im = cv2.resize(im_cpu, (newEnvWidth, newEnvHeight), interpolation = cv2.INTER_AREA )
    else:
        im = cv2.resize(im_cpu, (newEnvWidth, newEnvHeight), interpolation = cv2.INTER_LINEAR )


    im = (np.transpose(im, [2, 0, 1] ).astype(np.float32 ) / 255.0 )[np.newaxis, :, :, :]
    im = im / im.max()
    imBatchSmall = Variable(torch.from_numpy(im**(2.2) ) ).cuda()
    renderLayer = models.renderingLayer(isCuda = opt.cuda,
            imWidth=newEnvWidth, imHeight=newEnvHeight, fov = fov,
            envWidth = opt.envWidth, envHeight = opt.envHeight)

    output2env = models.output2env(isCuda = opt.cuda,
            envWidth = opt.envWidth, envHeight = opt.envHeight, SGNum = opt.SGNum )

    ########################################################
    # Build the cascade network architecture #
    albedoPreds, normalPreds, roughPreds, depthPreds = [], [], [], []
    albedoBSPreds, roughBSPreds, depthBSPreds = [], [], []
    envmapsPreds, envmapsPredImages = [], []

    ################# BRDF Prediction ######################
    inputBatch = imBatches[0]
    x1, x2, x3, x4, x5, x6 = encoders[0](inputBatch )

    albedoPred = 0.5 * (albedoDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6) + 1)
    normalPred = normalDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6)
    roughPred = roughDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6 )
    depthPred = 0.5 * (depthDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6) + 1)

    # Normalize Albedo and depth
    bn, ch, nrow, ncol = albedoPred.size()
    albedoPred = albedoPred.view(bn, -1)
    albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
    albedoPred = albedoPred.view(bn, ch, nrow, ncol)

    bn, ch, nrow, ncol = depthPred.size()
    depthPred = depthPred.view(bn, -1)
    depthPred = depthPred / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
    depthPred = depthPred.view(bn, ch, nrow, ncol)

    albedoPreds.append(albedoPred )
    normalPreds.append(normalPred )
    roughPreds.append(roughPred )
    depthPreds.append(depthPred )

    ################# Lighting Prediction ###################
    if opt.isLight or opt.level == 2:
        # Interpolation
        imBatchLarge = F.interpolate(imBatches[0], [imBatchSmall.size(2) *
            4, imBatchSmall.size(3) * 4], mode='bilinear')
        albedoPredLarge = F.interpolate(albedoPreds[0], [imBatchSmall.size(2)*
            4, imBatchSmall.size(3) * 4], mode='bilinear')
        normalPredLarge = F.interpolate(normalPreds[0], [imBatchSmall.size(2) *
            4, imBatchSmall.size(3) * 4], mode='bilinear')
        roughPredLarge = F.interpolate(roughPreds[0], [imBatchSmall.size(2) *
            4, imBatchSmall.size(3) * 4], mode='bilinear')
        depthPredLarge = F.interpolate(depthPreds[0], [imBatchSmall.size(2) *
            4, imBatchSmall.size(3) * 4], mode='bilinear')

        inputBatch = torch.cat([imBatchLarge, albedoPredLarge,
            0.5*(normalPredLarge+1), 0.5*(roughPredLarge+1), depthPredLarge ], dim=1 )
        x1, x2, x3, x4, x5, x6 = lightEncoders[0](inputBatch )

        # Prediction
        axisPred = axisDecoders[0](x1, x2, x3, x4, x5, x6, imBatchSmall )
        lambPred = lambDecoders[0](x1, x2, x3, x4, x5, x6, imBatchSmall )
        weightPred = weightDecoders[0](x1, x2, x3, x4, x5, x6, imBatchSmall )
        bn, SGNum, _, envRow, envCol = axisPred.size()
        envmapsPred = torch.cat([axisPred.view(bn, SGNum*3, envRow, envCol ), lambPred, weightPred], dim=1)
        envmapsPreds.append(envmapsPred )

        envmapsPredImage, axisPred, lambPred, weightPred = output2env.output2env(axisPred, lambPred, weightPred )
        envmapsPredImages.append(envmapsPredImage )

        diffusePred, specularPred = renderLayer.forwardEnv(albedoPreds[0], normalPreds[0],
                roughPreds[0], envmapsPredImages[0] )

        diffusePredNew, specularPredNew = models.LSregressDiffSpec(
                diffusePred,
                specularPred,
                imBatchSmall,
                diffusePred, specularPred )

        diffusePred = diffusePredNew
        specularPred = specularPredNew

    #################### BRDF Prediction ####################
    '''
    if opt.level == 2:
        albedoPredLarge = F.interpolate(albedoPreds[0], [newEnvHeight, newEnvWidth ], mode='bilinear')
        normalPredLarge = F.interpolate(normalPreds[0], [newEnvHeight, newEnvWidth ], mode='bilinear')
        roughPredLarge = F.interpolate(roughPreds[0], [newEnvHeight, newEnvWidth ], mode='bilinear')
        depthPredLarge = F.interpolate(depthPreds[0], [newEnvHeight, newEnvWidth ], mode='bilinear')

        diffusePredLarge = F.interpolate(diffusePred, [newEnvHeight, newEnvWidth ], mode='bilinear')
        specularPredLarge = F.interpolate(specularPred, [newEnvHeight, newEnvWidth ], mode='bilinear')

        inputBatch = torch.cat([imBatches[1], albedoPredLarge,
            0.5 * (normalPredLarge+1), 0.5*(roughPredLarge+1), depthPredLarge,
            diffusePredLarge, specularPredLarge], dim=1)

        x1, x2, x3, x4, x5, x6 = encoders[1](inputBatch )
        albedoPred = 0.5 * (albedoDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6) + 1)
        normalPred = normalDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6)
        roughPred = roughDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6 )
        depthPred = 0.5 * (depthDecoders[1](imBatches[1], x1, x2, x3, x4, x5, x6) + 1)

        # Normalize Albedo and depth
        bn, ch, nrow, ncol = albedoPred.size()
        albedoPred = albedoPred.view(bn, -1)
        albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        albedoPred = albedoPred.view(bn, ch, nrow, ncol)

        bn, ch, nrow, ncol = depthPred.size()
        depthPred = depthPred.view(bn, -1)
        depthPred = depthPred / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        depthPred = depthPred.view(bn, ch, nrow, ncol)

        albedoPreds.append(albedoPred )
        normalPreds.append(normalPred )
        roughPreds.append(roughPred )
        depthPreds.append(depthPred )

    ############### Lighting Prediction ######################
    if opt.level == 2 and opt.isLight:
        # Interpolation
        imBatchLarge = F.interpolate(imBatches[1], [imBatchSmall.size(2) *
            4, imBatchSmall.size(3) * 4], mode='bilinear')
        albedoPredLarge = F.interpolate(albedoPreds[1], [imBatchSmall.size(2)*
            4, imBatchSmall.size(3) * 4], mode='bilinear')
        normalPredLarge = F.interpolate(normalPreds[1], [imBatchSmall.size(2) *
            4, imBatchSmall.size(3) * 4], mode='bilinear')
        roughPredLarge = F.interpolate(roughPreds[1], [imBatchSmall.size(2) *
            4, imBatchSmall.size(3) * 4], mode='bilinear')
        depthPredLarge = F.interpolate(depthPreds[1], [imBatchSmall.size(2) *
            4, imBatchSmall.size(3) * 4], mode='bilinear')

        inputBatch = torch.cat([imBatchLarge, albedoPredLarge,
            0.5*(normalPredLarge+1), 0.5*(roughPredLarge+1), depthPredLarge ], dim=1 )
        x1, x2, x3, x4, x5, x6 = lightEncoders[1](inputBatch, envmapsPred )

        # Prediction
        axisPred = axisDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall )
        lambPred = lambDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall )
        weightPred = weightDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall )
        bn, SGNum, _, envRow, envCol = axisPred.size()
        envmapsPred = torch.cat([axisPred.view(bn, SGNum*3, envRow, envCol ), lambPred, weightPred], dim=1)
        envmapsPreds.append(envmapsPred )
    '''

    #################### Output Results #######################
    if opt.isLight:
        # Save the envmapImages
        for n in range(1, len(envmapsPreds ) ):
            envmapsPred = cupy.asarray(envmapsPreds[n])#envmapsPreds[n].data.cpu().numpy()
            shading = utils.predToShading(envmapsPred, SGNum = opt.SGNum )
            shading = shading.transpose([1, 2, 0] )
            shading = shading / np.mean(shading ) / 3.0
            shading = np.clip(shading, 0, 1)
            shading = (255 * shading ** (1.0/2.2) ).astype(np.uint8 )
            cv2.imwrite(shadingNames[n], shading[:, :, ::-1] )
    print("Processed image {}".format(time.time()-procimgstarttick))
print("Total Time {}".format(time.time()-starttick))

