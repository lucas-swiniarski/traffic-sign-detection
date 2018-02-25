"""
Implement a small convnet with 3 convolutions and 2 linear layers on 48-dimensional images.
"""

local nn = require 'nn'

-- Build model with cudnn Convolution if necessary :
-- 48x48 entry

function build_model(libs)
  local SpatialConvolution = libs['SpatialConvolution']
  local SpatialMaxPooling = libs['SpatialMaxPooling']
  local ReLU = libs['ReLU']

  local model  = nn.Sequential()

  model:add(SpatialConvolution(3, 100, 7, 7)) -- 100 x 42 x 42
  model:add(SpatialBatchNormalization(100, 1e-3, nil, true))
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2)) -- 100 x 21 x 21
  model:add(SpatialConvolution(100, 150, 4, 4)) -- 150 x 18 x 18
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2)) -- 150 x 9 x 9
  model:add(SpatialConvolution(150, 250, 4, 4)) -- 250 x 6 x 6
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2)) -- 250 x 3 x 3
  model:add(nn.View(2250))
  model:add(nn.Linear(2250, 300))
  model:add(nn.ReLU())
  model:add(nn.Linear(300, 43))

  return model
end
