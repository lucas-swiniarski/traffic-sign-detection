require 'math'
require 'torch'

local tnt = require 'torchnet'
local image = require 'image'
local utils = require 'utils'

local trainData = torch.load(DATA_PATH ..'train.t7')
local testData = torch.load(DATA_PATH ..'test.t7')

local DATA_PATH_IN = "./data/"

torch.setdefaulttensortype('torch.DoubleTensor')

allImages = torch.DoubleTensor(trainData:size(1) + testData:size(1), 3, 48, 48)
for i = 1, trainData:size(1) do
  allImages[i] = getTrainSample(trainData, i, DATA_PATH_IN, 0, 48, 48, false)
end

for i = 1, testData:size(1) do
  allImages[i] = getTrainSample(trainData, i, DATA_PATH_IN, 0, 48, 48, 1)
-- Load Training set
