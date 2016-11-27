local nn = require 'nn'

-- Build model with cudnn Convolution if necessary :
-- 48x48 entry

function build_model(libs)
  local SpatialConvolution = libs['SpatialConvolution']
  local SpatialMaxPooling = libs['SpatialMaxPooling']
  local ReLU = libs['ReLU']

  local model  = nn.Sequential()

  model:add(SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1)) -- 16 x 48 x 48
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2)) -- 16 x 24 x 24
  model:add(SpatialConvolution(16, 32, 5, 5, 1, 1, 2, 2)) -- 32 x 24 x 24
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2)) -- 32 x 12 x 12
  model:add(SpatialConvolution(32, 64, 7, 7, 1, 1, 3, 3)) -- 64 x 12 x 12
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2)) -- 64 x 6 x 6
  model:add(SpatialConvolution(64, 64, 11, 11, 1, 1, 5, 5)) -- 64 x 12 x 12
  model:add(ReLU())
  model:add(nn.View(9216))
  model:add(nn.Linear(1024, 86))
  model:add(nn.ReLU())
  model:add(nn.Linear(86, 43))

  return model
end
