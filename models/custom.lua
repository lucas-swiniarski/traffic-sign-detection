local nn = require 'nn'

-- Build model with cudnn Convolution if necessary :
-- 32x32 entry

function build_model(libs)
  local SpatialConvolution = libs['SpatialConvolution']
  local SpatialMaxPooling = libs['SpatialMaxPooling']
  local ReLU = libs['ReLU']

  local model  = nn.Sequential()

  model:add(SpatialConvolution(3, 16, 5, 5))
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2,2,2))
  model:add(SpatialConvolution(16, 128, 5, 5))
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2,2,2))
  model:add(nn.View(3200))
  model:add(nn.Linear(3200, 64))
  model:add(nn.ReLU())
  model:add(nn.Linear(64, 43))

  return model
end
