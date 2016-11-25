require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'

--[[
--  Hint:  Plot as much as you can.
--  Look into torch wiki for packages that can help you plot.
--]]

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 32, 32
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

torch.setdefaulttensortype('torch.DoubleTensor')

-- torch.setnumthreads(1)
torch.manualSeed(opt.manualSeed)
-- cutorch.manualSeedAll(opt.manualSeed)

function resize(img)
    return image.scale(img, WIDTH,HEIGHT)
end

--[[
-- Hint:  Should we add some more transforms? shifting, scaling?
-- Should all images be of size 32x32?  Are we losing
-- information by resizing bigger images to a smaller size?
--]]
function transformInput(inp)
    f = tnt.transform.compose{
        [1] = resize
    }
    return f(inp)
end

function getTrainSample(dataset, idx)
  r = dataset[idx]
  classId, track, file = r[9], r[1], r[2]
  file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
  return transformInput(image.load(DATA_PATH .. '/train_images/'..file))
end

function getTrainLabel(dataset, idx)
  return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
  r = dataset[idx]
  file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
  return transformInput(image.load(file))
end

function getIterator(dataset)
  local d = tnt.BatchDataset{
      batchsize = opt.batchsize,
      dataset = dataset
  }

  d = d:shuffle() -- before the iterator()

  -- So the we can call 'iterator:exec('manualSeed', seed)'
  function d:manualSeed(seed) torch.manualSeed(opt.manualSeed) end

  return tnt.ParallelDatasetIterator{
    nthread = opt.nThreads,
    init = function() require 'torchnet' end,
    closure = function()
      return d
    end
  }
end

local trainData = torch.load(DATA_PATH..'train.t7')
local testData = torch.load(DATA_PATH..'test.t7')

trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    initialpartition = 'train',
    --[[
    --  Hint:  Use a resampling strategy that keeps the
    --  class distribution even during initial training epochs
    --  and then slowly converges to the actual distribution
    --  in later stages of training.
    --]]
    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, trainData:size(1)):long(),
            load = function(idx)
                return {
                    input =  getTrainSample(trainData, idx),
                    target = getTrainLabel(trainData, idx)
                }
            end
        }
    }
}

testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            sampleId = torch.LongTensor{testData[idx][1]}
        }
    end
}

-- If cudnn, get the fast convolutions
libs = {}

torch.setdefaulttensortype('torch.FloatTensor')

if opt.cudnn then
    print("using cudnn")
    require 'cudnn'
    require 'cunn'
    libs['SpatialConvolution'] = cudnn.SpatialConvolution
    libs['SpatialMaxPooling'] = cudnn.SpatialMaxPooling
    libs['ReLU'] = cudnn.ReLU
else
    libs['SpatialConvolution'] = nn.SpatialConvolution
    libs['SpatialMaxPooling'] = nn.SpatialMaxPooling
    libs['ReLU'] = nn.ReLU
end

require("models/" .. opt.model)

local model = build_model(libs)
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1

if opt.cudnn then
  model = model:cuda()
  criterion = criterion:cuda()
end

-- print(model)

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end


-- Cuda for input / label
engine.hooks.onSample = function(state)
  if opt.cudnn then
    state.sample.input = state.sample.input:cuda()
    state.sample.target = state.sample.target:cuda()
  else
    state.sample.input = state.sample.input:float()
  end
end

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, state.iterator.dataset:size(), meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

-- After each epoch, if not learning : Divide by 2 learning rate.

local best_val = nil

engine.hooks.onEnd = function(state)

    if mode == "Val" then

      local val_err = clerr:value{k=1}

      if best_val == nil then
        best_val = val_err
      end

      if val_err > best_val then
        opt.LR = opt.LR / 2
        print("Not the best validation. Best so far : ", best_val, "New lr : ", opt.LR)
      else
        best_val = val_err
      end
    end

    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
end

local epoch = 1

while epoch <= opt.nEpochs do
  print("Epoch : " .. epoch)
  trainDataset:select('train')
  engine:train{
      network = model,
      criterion = criterion,
      iterator = getIterator(trainDataset),
      optimMethod = optim.sgd,
      maxepoch = 1,
      config = {
          learningRate = opt.LR,
          momentum = opt.momentum
      }
  }

  trainDataset:select('val')
  engine:test{
      network = model,
      criterion = criterion,
      iterator = getIterator(trainDataset)
  }

  print('Done with Epoch '..tostring(epoch))
  epoch = epoch + 1

  trainDataset:select('train')
  trainDataset:exec('manualSeed', epoch)
  trainDataset:exec('resample')
end

-- Do
-- Not
-- Change
-- Anything
-- Here

local submission = assert(io.open(opt.logDir .. "/submission.csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]
engine.hooks.onForward = function(state)
  local fileNames  = state.sample.sampleId
  local _, pred = state.network.output:max(2)
  pred = pred - 1
  for i = 1, pred:size(1) do
      submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
  end
  xlua.progress(batch, state.iterator.dataset:size())
  batch = batch + 1
end

engine.hooks.onEnd = function(state)
  submission:close()
end

engine:test{
    network = model,
    iterator = getIterator(testDataset)
}

print("The End!")
