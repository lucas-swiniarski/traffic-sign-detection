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

print(' Loading train images ... ')

allImages = torch.DoubleTensor(trainData:size(1) + testData:size(1), 3, 48, 48)
for i = 1, sizeTrainData do
  allImages[i] = getTrainSample(trainData, i, DATA_PATH_IN, 0, 48, 48, false)
  xlua.progress(i, sizeTrainData)
end

print(' Loading test images ... ')

for i = 1, sizeTestData do
  allImages[sizeTrainData + i] = getTestSample(trainData, i, DATA_PATH_IN, 0, 48, 48, 1)
  xlua.progress(i, sizeTestData)
end

print(' Remove mean ... ')
-- Remove mean
allImages = allImages:csub(torch.mean(allImages))

print(' Remove std ... ')
-- Remove std
allImages = allImages:reshape(sizeTrainData + sizeTestData, 3 * 48 * 48)
allImages = allImages:div(torch.std(allImages:add(10):sqrt(), 2))
allImages = allImages:reshape(sizeTrainData + sizeTestData, 3, 48, 48)

print(' Create dirs ... ')
paths.rmall(DATA_PATH_OUT)
paths.mkdir(DATA_PATH_OUT)

paths.mkdir(DATA_PATH_OUT .. 'train_images')
paths.mkdir(DATA_PATH_OUT .. 'test_images')

for i = 0, 42 do
  s = '000'
  if i < 10 then
    s = s .. '0'
  end
  s = s .. string(i)
  paths.mkdir(DATA_PATH_OUT .. 'train_images/' .. s)
end

print(' Save training set ... ')

for i = 1, sizeTrainData do
  r = trainData[idx]
  classId, track, file = r[9], r[1], r[2]
  file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
  image.save(DATA_PATH_OUT .. 'train_images/' .. file, allImages[i])
  xlua.progress(i, sizeTrainData)
end

print(' Save testing set ... ')

for i = 1, sizeTestData do
  r = dataset[idx]
  file = DATA_PATH_OUT .. "test_images/" .. string.format("%05d.ppm", r[1])
  image.save(file, allImages[sizeTrainData + i])
  xlua.progress(i, sizeTestData)
end
