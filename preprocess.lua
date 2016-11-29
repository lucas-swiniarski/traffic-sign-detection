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
  if i >= 100 then
    break
  end
end
print(' ')

print(' Loading test images ... ')

local testImages = torch.DoubleTensor(sizeTestData, 3, 48, 48)

for i = 1, sizeTestData do
  testImages[i] = getTestSample(trainData, i, DATA_PATH_IN, 48, 48)
  xlua.progress(i, sizeTestData)
  if i > 100 then
    break
  end
end

trainImagesResult = trainImages:copy(trainImages)
testImagesResult = testImages:copy(testImages)


print(' Remove mean ... ')
-- Remove mean
mean = trainImages:mean(2)

trainImagesResult:add(mean:mul(-1):view(trainImages:size(1),1):expandAs(trainImages))
testImagesResult:add(mean:mul(-1):view(testImages:size(1),1):expandAs(testImages))

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
