local nn = require 'nn'

-- Build model with cudnn Convolution if necessary :
-- 32x32 entry

function build_model(libs)
  local SpatialConvolution = libs['SpatialConvolution']
  local SpatialMaxPooling = libs['SpatialMaxPooling']
  local ReLU = libs['ReLU']

  local model  = nn.Sequential()

  model:add(SpatialConvolution(3, 16, 5, 5)) --16x28x28
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2,2,2))-- 16x32x32 -> 16x7x7
  model:add(SpatialConvolution(16, 128, 5, 5)) -- 128x5x
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2,2,2))-- 128x5x5
  model:add(nn.View(3200))
  model:add(nn.Linear(3200, 64))
  model:add(nn.ReLU())
  model:add(nn.Linear(64, 43))

  return model
end
