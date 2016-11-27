require 'math'
require 'torch'

local tnt = require 'torchnet'
local image = require 'image'

local WIDTH, HEIGHT = 32, 32

function resize(img, theta_max)
  return image.scale(img, WIDTH,HEIGHT, 'bicubic')
end

function rotate(img, theta_max)
  return image.rotate(img, torch.uniform(- theta_max, theta_max), 'bilinear')
end


--[[
-- Hint:  Should we add some more transforms? shifting, scaling?
-- Should all images be of size 32x32?  Are we losing
-- information by resizing bigger images to a smaller size?
--]]
function transformInput(inp, theta_max)
  theta_max = theta_max
  f = tnt.transform.compose{
      [1] = function(img) return image.rotate(img, torch.uniform(- theta_max, theta_max), 'bilinear') end,
      [2] = resize
  }
  return f(inp)
end

function tranformInputTest(inp)
  f = tnt.transform.compose{
      [1] = resize
  }
  return f(inp)
end

function getTrainSample(dataset, idx, DATA_PATH, theta_max)
  r = dataset[idx]
  classId, track, file = r[9], r[1], r[2]
  file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
  return transformInput(image.load(DATA_PATH .. '/train_images/'..file), theta_max)
end

function getTrainLabel(dataset, idx)
  return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx, DATA_PATH)
  r = dataset[idx]
  file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
  return tranformInputTest(image.load(file))
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
