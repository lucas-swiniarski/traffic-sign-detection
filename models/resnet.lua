local nn = require 'nn'
local cudnn = require 'cudnn'

-- Build model with cudnn Convolution if necessary :
-- 48x48 entry

function build_model(libs)
  local SpatialConvolution = libs['SpatialConvolution']
  local SpatialMaxPooling = libs['SpatialMaxPooling']
  local SpatialBatchNormalization = cudnn.SpatialBatchNormalization
  local ReLU = libs['ReLU']

  local function ConvBN(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
      local module = nn.Sequential()
      module:add(SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
      module:add(SpatialBatchNormalization(nOutputPlane,1e-3,nil,true))
      module:add(ReLU(true))
      return module
  end

  local function Residual(m)
    local module = nn.Sequential()
    local cat = nn.ConcatTable():add(nn.Identity()):add(m)
    module:add(cat):add(nn.CAddTable())
    return module
  end

  local model = nn.Sequential()
  model:add(ConvBN(3, 100, 7, 7, 1, 1, 1, 1)) --> 100 x 42 x 42

  model:add(SpatialMaxPooling(2,2)) --> 100 x 21 x 21

  local cat1 = nn.Sequential()
  cat1:add(ConvBN(100, 100, 3, 5, 1, 1, 1, 1))
  cat1:add(ConvBN(100, 100, 5, 5, 1, 1, 2, 2))
  cat1:add(ConvBN(100, 100, 7, 5, 1, 1, 3, 3))

  model:add(Residual(cat1)) --> 100 x 21 x 21

  model:add(ConvBN(100, 150, 4, 4)) --> 150 x 18 x 18
  model:add(SpatialMaxPooling(2,2)) --> 150 x 9 x 9

  local cat2 = nn.Sequential()
  cat2:add(ConvBN(150, 150, 3, 3, 1, 1, 1, 1))
  cat2:add(ConvBN(150, 150, 5, 5, 1, 1, 2, 2))
  cat2:add(ConvBN(150, 150, 7, 7, 1, 1, 3, 3))

  model:add(Residual(cat2)) --> 150 x 9 x 9

  model:add(SpatialConvolution(150, 250, 4, 4)) -- 250 x 6 x 6
  model:add(ReLU())
  model:add(SpatialMaxPooling(2,2)) --> 250 x 3 x 3

  local cat3 = nn.Sequential()
  cat3:add(ConvBN(250, 250, 3, 3, 1, 1, 1, 1))
  cat3:add(ConvBN(250, 250, 5, 5, 1, 1, 2, 2))
  cat3:add(ConvBN(250, 250, 7, 7, 1, 1, 3, 3))

  model:add(Residual(cat3))

  model:add(nn.View(2250))
  model:add(nn.Linear(2250, 300))
  model:add(nn.ReLU())
  model:add(nn.Linear(300, 43))

  return model
end
