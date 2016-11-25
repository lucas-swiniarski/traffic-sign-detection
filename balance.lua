require 'data_prep.labels'

local tnt = require 'torchnet'

local paintingsPerArtists = 100
local minPaintingsPerArtists = 10

local function getArtistImages(imageIds)
  local labels = getLabelsFromCsv();

  local artistImages = {}

  local label

  for imageIndex=1,table.getn(imageIds) do
    local imageId = imageIds[imageIndex]
    label = labels[imageId]

    if artistImages[label.artistKey] == nil then
      artistImages[label.artistKey] = {imageIndex}
    else
      artistImages[label.artistKey][table.getn(artistImages[label.artistKey])+1] = imageIndex
    end
  end

  return artistImages
end

function balanceArtists(dataset, imageIds)
  local artistImages = getArtistImages(imageIds)

  for i,list in pairs(artistImages) do
    if table.getn(artistImages[i]) < minPaintingsPerArtists then
      artistImages.i = nil
    end
  end

  --- For each artist, upsample to 40 paintings:
  for i,list in pairs(artistImages) do
    if table.getn(artistImages[i]) < paintingsPerArtists then
      while table.getn(artistImages[i]) < paintingsPerArtists do
        artistImages[i][#artistImages[i]+1] = artistImages[i][math.random(#artistImages[i])]
      end
    else
      -- If artist > 40 images, first 40 images ...
      artistImages[i] = {unpack(artistImages[i], 1, paintingsPerArtists)}
    end
  end

  local rebalancedImageIndeces = {}

  -- flatten into list of indeces
  for i,imageList in pairs(artistImages) do
    for j,imageIndex in pairs(imageList) do
      table.insert(rebalancedImageIndeces, imageIndex)
    end
  end

  local shuffle = torch.randperm(table.getn(rebalancedImageIndeces))

  print('number of images after rebalance:' .. #rebalancedImageIndeces)

  return tnt.ResampleDataset{
    dataset = dataset,
    size = table.getn(rebalancedImageIndeces),
    sampler = function(dataset, idx)
      return rebalancedImageIndeces[shuffle[idx]]
    end
  }
end
