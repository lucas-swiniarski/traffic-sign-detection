local nn = require 'nn'

-- Build model with cudnn Convolution if necessary :
-- 48x48 entry

function build_model(libs)
  local SpatialConvolution = libs['SpatialConvolution']
  local SpatialMaxPooling = libs['SpatialMaxPooling']
  local ReLU = libs['ReLU']

  local model  = nn.Sequential()

  model:add(SpatialConvolution(3, 32, 7, 7)) -- 32 x 42 x 42
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2)) -- 32 x 21 x 21
  model:add(SpatialConvolution(32, 64, 4, 4)) -- 64 x 18 x 18
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2)) -- 64 x 9 x 9
  model:add(SpatialConvolution(64, 128, 4, 4)) -- 128 x 6 x 6
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2)) -- 128 x 3 x 3
  model:add(nn.View(1152))
  model:add(nn.Linear(1152, 215))
  model:add(nn.ReLU())
  model:add(nn.Linear(215, 43))

  return model
end
