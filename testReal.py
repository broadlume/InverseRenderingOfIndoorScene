import torch
import numpy as np
import cupy as cp
from torch.autograd import Variable
import argparse
import random
import os
import models
import utils
import os.path as osp
import cv2
import torch.nn.functional as F
import utils

from InvRenderModels import BrdfModel0, BrdfModel1

from pytictoc import TicToc

tictoc = TicToc();

inputWidth = 320
inputHeight = 240


parser = argparse.ArgumentParser()
# The locationi of testing set
parser.add_argument('--dataRoot', default='../datasets/test_images_png', help='path to real images')
parser.add_argument('--imList', default='imList_20.txt', help='path to image list')

parser.add_argument('--testRoot', default='Real20', help='the path to save the testing errors' )

parser.add_argument('--envRow', type=int, default=120, help='the height /width of the envmap predictions')
parser.add_argument('--envCol', type=int, default=160, help='the height /width of the envmap predictions')
parser.add_argument('--envHeight', type=int, default=8, help='the height /width of the envmap predictions')
parser.add_argument('--envWidth', type=int, default=16, help='the height /width of the envmap predictions')

parser.add_argument('--SGNum', type=int, default=12, help='the number of spherical Gaussian lobes')
parser.add_argument('--offset', type=float, default=1, help='the offset when train the lighting network')

parser.add_argument('--cuda', default=True, action = 'store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for testing network')

parser.add_argument('--level', type=int, default=2, help='the cascade level')
parser.add_argument('--isLight', default=True, action='store_true', help='whether to predict lightig')

# Image Picking
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]
device = 'cpu'
if opt.cuda:
    device = 'cuda:{0}'.format(opt.gpuId)

experiments = ['models/check_cascade0_w320_h240', 'models/check_cascade1_w320_h240' ]
experimentsLight = ['models/check_cascadeLight0_sg12_offset1', 'models/check_cascadeLight1_sg12_offset1' ]
nepochs = [14, 7]
nepochsLight = [10, 10 ]

imHeights = [inputHeight, inputHeight ]
imWidths = [inputWidth, inputWidth ]

os.system('mkdir {0}'.format(opt.testRoot ) )
os.system('cp *.py %s' % opt.testRoot )

opt.seed = 0
random.seed(opt.seed )
torch.manual_seed(opt.seed )

opt.batchSize = 1


brdfModel0 = BrdfModel0(device)
brdfModel1 = BrdfModel1(device)

lightEncoders= []
axisDecoders = []
lambDecoders = []
weightDecoders = []

tictoc.tic()

imBatchSmall = Variable(torch.FloatTensor(opt.batchSize, 3, opt.envRow, opt.envCol ) )
for n in range(0, opt.level ):
    if opt.isLight or (opt.level == 2 and n == 0):
        # Light network
        lightEncoders.append(models.encoderLight(cascadeLevel = n, SGNum = opt.SGNum).eval().to(device) )
        axisDecoders.append(models.decoderLight(mode=0, SGNum = opt.SGNum ).eval().to(device) )
        lambDecoders.append(models.decoderLight(mode=1, SGNum = opt.SGNum ).eval().to(device) )
        weightDecoders.append(models.decoderLight(mode=2, SGNum = opt.SGNum ).eval().to(device) )

        lightEncoders[n].load_state_dict(
                torch.load('{0}/lightEncoder{1}_{2}.pth'.format(experimentsLight[n], n, nepochsLight[n]-1) ).state_dict() )
        axisDecoders[n].load_state_dict(
                torch.load('{0}/axisDecoder{1}_{2}.pth'.format(experimentsLight[n], n, nepochsLight[n]-1) ).state_dict() )
        lambDecoders[n].load_state_dict(
                torch.load('{0}/lambDecoder{1}_{2}.pth'.format(experimentsLight[n], n, nepochsLight[n]-1) ).state_dict() )
        weightDecoders[n].load_state_dict(
                torch.load('{0}/weightDecoder{1}_{2}.pth'.format(experimentsLight[n], n, nepochsLight[n]-1) ).state_dict() )

#########################################

print("Loaded models")
tictoc.toc()
tictoc.tic()



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
    j += 1
    print('%d/%d: %s' % (j, len(imList), imName) )

    imBatches = []

    albedoImNames = []
    normalImNames = []
    roughImNames = []
    depthImNames = []
    imOutputNames = []
    envmapPredImNames = []
    renderedImNames = []
    shadingNames, envmapsPredSGNames = [], []

    imId = imName.split('/')[-1]
    print(imId )
    imOutputNames.append(osp.join(outfilename, imId ) )

    for n in range(0, opt.level ):
        albedoImNames.append(osp.join(outfilename, imId.replace('.png', '_albedo%d.png' % n ) ) )
        normalImNames.append(osp.join(outfilename, imId.replace('.png', '_normal%d.png' % n) ) )
        roughImNames.append(osp.join(outfilename, imId.replace('.png', '_rough%d.png' % n) ) )
        depthImNames.append(osp.join(outfilename, imId.replace('.png', '_depth%d.png' % n) ) )

        shadingNames.append(osp.join(outfilename, imId.replace('.png', '_shading%d.png' % n) ) )
        envmapPredImNames.append(osp.join(outfilename, imId.replace('.png', '_envmap%d.png' % n) ) )
        renderedImNames.append(osp.join(outfilename, imId.replace('.png', '_rendered%d.png' % n) ) )

    # Load the image from cpu to gpu
    assert(osp.isfile(imName ) )
    im_cpu = cv2.imread(imName )[:, :, ::-1]
    nh, nw = im_cpu.shape[0], im_cpu.shape[1]

    # Resize Input Images
    newImWidth = []
    newImHeight = []
    for n in range(0, opt.level ):
        if nh < nw:
            newW = imWidths[n]
            newH = int(float(imWidths[n] ) / float(nw) * nh )
        else:
            newH = imHeights[n]
            newW = int(float(imHeights[n] ) / float(nh) * nw )

        if nh < newH:
            im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_AREA )
        else:
            im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_LINEAR )

        newImWidth.append(newW )
        newImHeight.append(newH )

        im = (np.transpose(im, [2, 0, 1] ).astype(np.float32 ) / 255.0 )[np.newaxis, :, :, :]
        im = im / im.max()
        imBatches.append( Variable(torch.from_numpy(im**(2.2) ) ).to(device) )

    nh, nw = newImHeight[-1], newImWidth[-1]

    newEnvWidth, newEnvHeight, fov = 0, 0, 0
    if nh < nw:
        fov = 57
        newW = opt.envCol
        newH = int(float(opt.envCol ) / float(nw) * nh )
    else:
        fov = 42.75
        newH = opt.envRow
        newW = int(float(opt.envRow ) / float(nh) * nw )

    if nh < newH:
        im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_AREA )
    else:
        im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_LINEAR )

    newEnvWidth = newW
    newEnvHeight = newH

    im = (np.transpose(im, [2, 0, 1] ).astype(np.float32 ) / 255.0 )[np.newaxis, :, :, :]
    im = im / im.max()
    imBatchSmall = Variable(torch.from_numpy(im**(2.2) ) ).to(device)
    renderLayer = models.renderingLayer(isCuda = opt.cuda,
            imWidth=newEnvWidth, imHeight=newEnvHeight, fov = fov,
            envWidth = opt.envWidth, envHeight = opt.envHeight)

    output2env = models.output2env(isCuda = opt.cuda,
            envWidth = opt.envWidth, envHeight = opt.envHeight, SGNum = opt.SGNum )

    ########################################################
    # Build the cascade network architecture #
    albedoPreds, normalPreds, roughPreds, depthPreds = [], [], [], []
    envmapsPreds, envmapsPredImages, renderedPreds = [], [], []
    cAlbedos = []
    cLights = []

    ################# BRDF Prediction ######################
    albedoPred, normalPred, roughPred, depthPred = brdfModel0(imBatches[0])

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

        with torch.no_grad():
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
        renderedPred = diffusePredNew + specularPredNew
        renderedPreds.append(renderedPred )

        cDiff, cSpec = (torch.sum(diffusePredNew) / torch.sum(diffusePred )).data.item(), ((torch.sum(specularPredNew) ) / (torch.sum(specularPred) ) ).data.item()
        if cSpec < 1e-3:
            cAlbedo = 1/ albedoPreds[-1].max().data.item()
            cLight = cDiff / cAlbedo
        else:
            cLight = cSpec
            cAlbedo = cDiff / cLight
            cAlbedo = np.clip(cAlbedo, 1e-3, 1 / albedoPreds[-1].max().data.item() )
            cLight = cDiff / cAlbedo
        envmapsPredImages[0] = envmapsPredImages[0] * cLight
        cAlbedos.append(cAlbedo )
        cLights.append(cLight )

        diffusePred = diffusePredNew
        specularPred = specularPredNew

    #################### BRDF Prediction ####################
    if opt.level == 2:
        diffusePredLarge = F.interpolate(diffusePred, [newImHeight[1], newImWidth[1] ], mode='bilinear')
        specularPredLarge = F.interpolate(specularPred, [newImHeight[1], newImWidth[1] ], mode='bilinear')
        
        albedoPred, normalPred, roughPred, depthPred = brdfModel1(imBatches[0], albedoPreds[0], normalPreds[0], roughPreds[0], depthPreds[0], diffusePredLarge, specularPredLarge)

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

        with torch.no_grad():
            x1, x2, x3, x4, x5, x6 = lightEncoders[1](inputBatch, envmapsPred )

            # Prediction
            axisPred = axisDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall )
            lambPred = lambDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall )
            weightPred = weightDecoders[1](x1, x2, x3, x4, x5, x6, imBatchSmall )

        bn, SGNum, _, envRow, envCol = axisPred.size()
        envmapsPred = torch.cat([axisPred.view(bn, SGNum*3, envRow, envCol ), lambPred, weightPred], dim=1)
        envmapsPreds.append(envmapsPred )

        envmapsPredImage, axisPred, lambPred, weightPred = output2env.output2env(axisPred, lambPred, weightPred )
        envmapsPredImages.append(envmapsPredImage )

        diffusePred, specularPred = renderLayer.forwardEnv(albedoPreds[1], normalPreds[1],
                roughPreds[1], envmapsPredImages[1] )

        diffusePredNew, specularPredNew = models.LSregressDiffSpec(
                diffusePred,
                specularPred,
                imBatchSmall,
                diffusePred, specularPred )

        renderedPre = diffusePredNew + specularPredNew
        renderedPreds.append(renderedPred )

        cDiff, cSpec = (torch.sum(diffusePredNew) / torch.sum(diffusePred)).data.item(), ((torch.sum(specularPredNew) ) / (torch.sum(specularPred) ) ).data.item()
        if cSpec == 0:
            cAlbedo = 1/ albedoPreds[-1].max().data.item()
            cLight = cDiff / cAlbedo
        else:
            cLight = cSpec
            cAlbedo = cDiff / cLight
            cAlbedo = np.clip(cAlbedo, 1e-3, 1 / albedoPreds[-1].max().data.item() )
            cLight = cDiff / cAlbedo
        envmapsPredImages[-1] = envmapsPredImages[-1] * cLight
        cAlbedos.append(cAlbedo )
        cLights.append(cLight )

        diffusePred = diffusePredNew
        specularPred = specularPredNew

    #################### Output Results #######################
    # Save the albedo
    for n in range(0, len(albedoPreds ) ):
        if n < len(cAlbedos ):
            albedoPred = (albedoPreds[n] * cAlbedos[n]).data.cpu().numpy().squeeze()
        else:
            albedoPred = albedoPreds[n].data.cpu().numpy().squeeze()

        albedoPred = albedoPred.transpose([1, 2, 0] )
        albedoPred = (albedoPred ) ** (1.0/2.2 )
        albedoPred = cv2.resize(albedoPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

        albedoPredIm = (np.clip(255 * albedoPred, 0, 255) ).astype(np.uint8)

        cv2.imwrite(albedoImNames[n], albedoPredIm[:, :, ::-1] )

    # Save the normal
    for n in range(0, len(normalPreds ) ):
        normalPred = normalPreds[n].data.cpu().numpy().squeeze()
        normalPred = normalPred.transpose([1, 2, 0] )
        normalPred = cv2.resize(normalPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

        normalPredIm = (255 * 0.5*(normalPred+1) ).astype(np.uint8)
        cv2.imwrite(normalImNames[n], normalPredIm[:, :, ::-1] )

    # Save the rough
    for n in range(0, len(roughPreds ) ):
        roughPred = roughPreds[n].data.cpu().numpy().squeeze()
        roughPred = cv2.resize(roughPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

        roughPredIm = (255 * 0.5*(roughPred+1) ).astype(np.uint8)
        cv2.imwrite(roughImNames[n], roughPredIm )

    # Save the depth
    for n in range(0, len(depthPreds ) ):
        depthPred = depthPreds[n].data.cpu().numpy().squeeze()

        depthPred = depthPred / np.maximum(depthPred.mean(), 1e-10) * 3
        depthPred = cv2.resize(depthPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

        depthOut = 1 / np.clip(depthPred+1, 1e-6, 10)
        depthPredIm = (255 * depthOut ).astype(np.uint8)
        cv2.imwrite(depthImNames[n], depthPredIm )

    if opt.isLight:
        for n in range(0, len(envmapsPreds ) ):
            envmapsPred = envmapsPreds[n].data.cpu().numpy()
            shading = utils.predToShading(cp.asarray(envmapsPred), SGNum = opt.SGNum )
            shading = shading.transpose([1, 2, 0] )
            shading = shading / np.mean(shading ) / 3.0
            shading = np.clip(shading, 0, 1)
            shading = (255 * shading ** (1.0/2.2) ).astype(np.uint8 )
            cv2.imwrite(shadingNames[n], shading[:, :, ::-1] )

        # Save the rendered image
        for n in range(0, len(renderedPreds ) ):
            renderedPred = renderedPreds[n].data.cpu().numpy().squeeze()
            renderedPred = renderedPred.transpose([1, 2, 0] )
            renderedPred = (renderedPred / renderedPred.max() ) ** (1.0/2.2)
            renderedPred = cv2.resize(renderedPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

            renderedPred = (np.clip(renderedPred, 0, 1) * 255).astype(np.uint8 )
            cv2.imwrite(renderedImNames[n], renderedPred[:, :, ::-1] )

    # Save the image
    cv2.imwrite(imOutputNames[0], im_cpu[:,:, ::-1] )

