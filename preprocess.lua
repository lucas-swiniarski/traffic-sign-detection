require 'math'
require 'torch'
require 'xlua'

local tnt = require 'torchnet'
local image = require 'image'
local utils = require 'utils'
local paths = require 'paths'

local DATA_PATH_IN = "./data/"
local DATA_PATH_OUT = "./data_scaled/"

local trainData = torch.load(DATA_PATH_IN ..'train.t7')
local testData = torch.load(DATA_PATH_IN ..'test.t7')

torch.setdefaulttensortype('torch.DoubleTensor')

local sizeTrainData = trainData:size(1)
local sizeTestData = testData:size(1)

print('Train data : ', sizeTrainData, ' test data : ', sizeTestData)

print(' Loading train images ... ')

local trainImages = torch.DoubleTensor(sizeTrainData, 3, 48, 48)

for i = 1, sizeTrainData do
  trainImages[i] = getTrainSample(trainData, i, DATA_PATH_IN, 0, 48, 48, false)
  xlua.progress(i, sizeTrainData)
end
print(' ')

print(' Loading test images ... ')

local testImages = torch.DoubleTensor(sizeTestData, 3, 48, 48)

for i = 1, sizeTestData do
  testImages[i] = getTestSample(trainData, i, DATA_PATH_IN, 48, 48)
  xlua.progress(i, sizeTestData)
end

local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1, sizeTrainData do
  xlua.progress(i, trainImages:size())
  -- rgb -> yuv
  local rgb = trainImages[i]
  local yuv = image.rgb2yuv(rgb)
  -- normalize y locally:
  yuv[1] = normalization(yuv[{{1}}])
  trainImages[i] = yuv
end
-- normalize u globally:
local mean_u = trainImages:select(2,2):mean()
local std_u = trainImages:select(2,2):std()
trainImages:select(2,2):add(-mean_u)
trainImages:select(2,2):div(std_u)
-- normalize v globally:
local mean_v = trainImages:select(2,3):mean()
local std_v = trainImages:select(2,3):std()
trainImages:select(2,3):add(-mean_v)
trainImages:select(2,3):div(std_v)

-- preprocess testSet
for i = 1, sizeTestData do
 xlua.progress(i, testData:size())
  -- rgb -> yuv
  local rgb = testImages[i]
  local yuv = image.rgb2yuv(rgb)
  -- normalize y locally:
  yuv[{1}] = normalization(yuv[{{1}}])
  testImages[i] = yuv
end
-- normalize u globally:
testImages:select(2,2):add(-mean_u)
testImages:select(2,2):div(std_u)
-- normalize v globally:
testImages:select(2,3):add(-mean_v)
testImages:select(2,3):div(std_v)

torch.save(DATA_PATH_OUT .. 'train_dataset.t7', trainImages)
torch.save(DATA_PATH_OUT .. 'test_dataset.t7', testImages)
