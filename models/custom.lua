local nn = require 'nn'

-- Build model with cudnn Convolution if necessary :
-- 32x32 entry

function build_model(libs)
  local SpatialConvolution = libs['SpatialConvolution']
  local SpatialMaxPooling = libs['SpatialMaxPooling']
  local ReLU = libs['ReLU']

  local model  = nn.Sequential()

  model:add(SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2)) -- 16 x 32 x 32
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2)) -- 16 x 16 x 16
  model:add(SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2)) -- 126 x 16 x 16
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2)) -- 128 x 8 x 8
  model:add(SpatialConvolution(64, 128, 11, 11, 1, 1, 5, 5)) -- 128 x 8 x 8
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2))
  model:add(nn.View(8192))
  model:add(nn.Linear(8192, 86))
  model:add(nn.ReLU())
  model:add(nn.Linear(86, 43))

  return model
end
