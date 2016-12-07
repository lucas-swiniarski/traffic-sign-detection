local nn = require 'nn'
local cudnn = require 'cudnn'

-- Build model with cudnn Convolution if necessary :
-- 52x52 entry

-- Best model so far, with 2 convolutions in each residual layer, size 5x5 kernel. 98.3 % accuracy on test set

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

  local function Residual2Conv(nInputPlane)
    local cat = nn.Sequential()
    module:add(ConvBN(nInputPlane, nInputPlane, 3, 3, 1, 1, 1, 1))
    module:add(ConvBN(nInputPlane, nInputPlane, 3, 3, 1, 1, 1, 1))
    return Residual(module)
  end

  local model = nn.Sequential()
  -- Input : 3 x 52 x 52
  model:add(ConvBN(3, 64, 7, 7, 2, 2, 2, 2)) --> 64 x 24 x 24

  model:add(Residual2Conv(64))
  model:add(Residual2Conv(64))
  model:add(Residual2Conv(64))

  model:add(ConvBN(64, 128, 3, 3, 2, 2, 1, 1)) --> 128 x 12 x 12

  model:add(Residual2Conv(128))
  model:add(Residual2Conv(128))
  model:add(Residual2Conv(128))

  model:add(ConvBN(128, 256, 3, 3, 2, 2, 1, 1)) --> 256 x 6 x 6

  model:add(Residual2Conv(256))
  model:add(Residual2Conv(256))
  model:add(Residual2Conv(256))

  model:add(ConvBN(256, 256, 3, 3, 2, 2, 1 ,1)) --> 256 x 3 x 3

  model:add(nn.View(2304))
  model:add(nn.Linear(2304, 300))
  model:add(nn.ReLU())
  model:add(nn.Linear(300, 43))

  return model
end
