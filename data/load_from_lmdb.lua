require 'cunn'
require 'cudnn'
opt = {                                                                                                                                                  
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 30,
   loadSize = 225,
   fineSize = 224,
   nThreads = 6,           -- #  of data loading threads to use
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
}
opt.data_classes_full = {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room',
                         'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'}
local feature_extractor = torch.load('/home/besedin/workspace/Models/ResNet_torch/resnet-200.t7')
feature_extractor:remove(14); feature_extractor:remove(13)
for idx_class = 1, #opt.data_classes_full do
  opt.data_classes = {opt.data_classes_full[idx_class]}
  local DataLoader = paths.dofile('./data.lua')
  local data = DataLoader.new(opt.nThreads, 'lsun', opt)
  local res = torch.CudaTensor(data:size(1), 2048)
  for idx_data = 1, data:size(1), opt.batchSize do
    xlua.progress(idx_data, data:size())
    batch = data:getBatch()
    batch = batch:cuda()
    if idx_data + opt.batchSize -1 <= data:size(1) then
      res[{{idx_data, idx_data + opt.batchSize - 1},{}}] = feature_extractor:forward(batch)
    else
      batch = batch[{{1, data:size(1)-idx_data+1},{},{},{}}]
      res[{{idx_data, data:size(1)},{}}] = feature_extractor:forward(batch)
      print('Last index = ' .. idx_data)
    end
  end
  torch.save(opt.data_classes[1] .. '_50k_features.t7', res:float())
end

