"""
Utility functions, for various treatments of train/test samples.
Functions for :
- Data Augmentation
- Centering the traffic signs.
"""
require 'math'
require 'torch'

local tnt = require 'torchnet'
local image = require 'image'

-- Transform Input Saturation, Brightness, Contrast !

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function Saturation(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function Brightness(var)
   local gs

   return function(input)
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function Contrast(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)
      gs:fill(gs[1]:mean())

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function transformInput(inp, theta_max, width, height)
  f = tnt.transform.compose{
    [1] = function(img) return Contrast(0.8)(img) end,
    [2] = function(img) return Saturation(0.8)(img) end,
    [3] = function(img) return Brightness(0.8)(img) end,
    [4] = function(img) return image.rotate(img, torch.uniform(- theta_max, theta_max), 'bilinear') end,
    [5] = function(img) return image.translate(img, torch.random(0, 11), torch.random(0, 11)) end,
    [6] = function(img) return image.scale(img, width + torch.random(-11, 11), height + torch.random(-11, 11)) end,
    [7] = function(img) return image.scale(img, width, height, 'bicubic') end
  }
  return f(inp)
end

function tranformInputTest(inp, width, height)
  f = tnt.transform.compose{
      [1] = function(img) return image.scale(img, width, height, 'bicubic') end
  }
  return f(inp)
end

function getTrainSample(dataset, idx, DATA_PATH, theta_max, width, height, isTraining)
  r = dataset[idx]
  classId, track, file = r[9], r[1], r[2]
  file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
  img = image.load(DATA_PATH .. '/train_images/'..file)
  if isTraining == true then
    return transformInput(image.crop(img, r[5], r[6], r[7], r[8]), theta_max, width, height)
  else
    return tranformInputTest(image.crop(img, r[5], r[6], r[7], r[8]), width, height)
  end
end

function getTrainLabel(dataset, idx)
  return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx, DATA_PATH, width, height)
  r = dataset[idx]
  file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
  img = image.load(file)
  return tranformInputTest(image.crop(img, r[4], r[5], r[6], r[7]), width, height)
end

function balanceTrainingSet(dataset, epoch, maxEpoch, trainData)
  -- Balance training dataset, less & less given epoch.
  local all_indexes = dataset.__dataset.__perm
  local class_indexes = {}

  for i = 1, dataset.__partitionsizes[1] do
    local label = getTrainLabel(trainData, all_indexes[i])[1]
    if class_indexes[label] ~= nil then
      table.insert(class_indexes[label], i)
    else
      class_indexes[label]= {i}
    end
  end

  local max = 0

  for class, image_list in pairs(class_indexes) do
    if table.getn(image_list) > max then
      max = table.getn(image_list)
    end
  end

  -- In order to slide toward the initial distribution when epochs near last we do :

  max = max * (maxEpoch - epoch + 1) / maxEpoch

  list_index_rebalanced = {}

  for class, image_list in pairs(class_indexes) do
    for i, image in pairs(image_list) do
      table.insert(list_index_rebalanced, image)
    end

    local image_inserted = table.getn(image_list)

    while image_inserted < max do
      table.insert(list_index_rebalanced, image_list[torch.random(#image_list)])
      image_inserted = image_inserted + 1
    end
  end

  local shuffle = torch.randperm(table.getn(list_index_rebalanced))

  return list_index_rebalanced, shuffle
end
