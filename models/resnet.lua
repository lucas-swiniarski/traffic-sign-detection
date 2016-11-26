local nn = require 'nn'
require './residual-layer'

-- Build model with cudnn Convolution if necessary :
-- 32x32 entry

function build_model(libs)
  local SpatialConvolution = libs['SpatialConvolution']
  local SpatialMaxPooling = libs['SpatialMaxPooling']
  local ReLU = libs['ReLU']

  local model  = nn.Sequential()

  model:add(SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1)) -- 16 x 32 x 32
  model:add(ReLU())

  model = addResidualLayer2(model, 16) -- 16 x 32 x 32
  model = addResidualLayer2(model, 16, 32, 2) -- 32 x 16 x 16

  model = addResidualLayer2(model, 32) -- 32 x 16 x 16
  model = addResidualLayer2(model, 32, 64, 2) -- 64 x 8 x 8

  model = addResidualLayer2(model, 64)

  model:add(nn.View(4096))
  model:add(nn.Linear(4096, 64))
  model:add(nn.ReLU())
  model:add(nn.Linear(64, 43))

  return model
end
