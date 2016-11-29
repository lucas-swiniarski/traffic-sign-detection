local nn = require 'nn'
local cudnn = require 'cudnn'

-- Build model with cudnn Convolution if necessary :
-- 48x48 entry

function build_model(libs)
  local SpatialConvolution = libs['SpatialConvolution']
  local SpatialMaxPooling = libs['SpatialMaxPooling']
  local SpatialBatchNormalization = cudnn.SpatialBatchNormalization
  local SpatialAveragePooling = cudnn.SpatialAveragePooling
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

  local function ConvConvResidual(nInputPlane)
    local cat = nn.Sequential()
    cat:add(ConvBN(nInputPlane, nInputPlane, 3, 3, 1, 1, 1, 1))
    cat:add(ConvBN(nInputPlane, nInputPlane, 3, 3, 1, 1, 1, 1))
    return Residual(cat)
  end

  local model = nn.Sequential() --> 3 x 48 x 48
  model:add(ConvBN(3, 64, 7, 7, 1, 1, 1, 1)) --> 100 x 42 x 42

  model:add(ConvBN(64, 64, 3, 3, 2, 2, 1, 1)) --> 100 x 21 x 21

  model:add(ConvConvResidual(64))
  model:add(ConvConvResidual(64))
  model:add(ConvConvResidual(64))

  model:add(ConvBN(64, 128, 4, 4)) --> 128 x 18 x 18
  model:add(ConvBN(128, 128, 3, 3, 2, 2, 1, 1)) --> 128 x 9 x 9

  model:add(ConvConvResidual(128))
  model:add(ConvConvResidual(128))
  model:add(ConvConvResidual(128))

  model:add(ConvBN(128, 256, 4, 4)) -- 250 x 6 x 6
  model:add(ConvBN(256, 256, 3, 3, 2, 2, 1, 1)) --> 256 x 4 x 4

  model:add(ConvConvResidual(256))
  model:add(ConvConvResidual(256))
  model:add(ConvConvResidual(256))

  model:add(SpatialAveragePooling(2, 2)) --> 256 x 2 x 2

  model:add(nn.View(1024))
  model:add(nn.Linear(1024, 300))
  model:add(nn.ReLU())
  model:add(nn.Linear(300, 43))

  return model
end
