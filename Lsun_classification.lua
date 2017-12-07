-- run with DATA_ROOT=/home/besedin/workspace/Data/LSUN/ls
require 'cunn'
require 'cudnn'
require 'image'
require 'optim'
require 'nngraph'

opt = {
  lr = 0.001,
  beta1 = 0.9,
  beta2 = 0.999,
  epsilon = 1e-8,
  batchSize = 500,
  fineSize = 224,
  gpu = 1,
  niter = 3,
  exp_nb = 3,
  dropout = 0,
  training = 'fake',
  testing = 'real',
}

opt.manualSeed = torch.random(1, 10000) -- fix seed
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(1)

local function getBatch(data, indices)
  local batch = {}
  batch.data = data.data:index(1, indices:long())
  batch.labels = data.labels:index(1, indices:long())
  return batch
end

local function load_dataset(data_name)
  local data_folder = '/home/besedin/workspace/Data/LSUN/single_file_t7/'
  trainset = torch.load(data_folder .. opt.training .. '/' .. data_name  .. '_train.t7')            
  trainset.data = trainset.data:cuda(); trainset.labels = trainset.labels:cuda()
  opt.trainSize = trainset.labels:size(1)
  testset = torch.load(data_folder .. opt.testing .. '/' .. data_name  .. '_test.t7')
  testset.data = testset.data:cuda(); testset.labels = testset.labels:cuda()
  opt.testSize = testset.labels:size(1)
  return trainset, testset
end

--[[local data_name = {
  'ResNet-18_224x224', 'ResNet-34_224x224', 'ResNet-50_224x224', 
  'ResNet-101_224x224' , 'ResNet-152_224x224','ResNet-200_224x224',
  'ResNet-18_128x128', 'ResNet-34_128x128', 'ResNet-50_128x128',
  'ResNet-101_128x128', 'ResNet-152_128x128', 'ResNet-200_128x128',
  'ResNet-18_64x64', 'ResNet-34_64x64', 'ResNet-50_64x64',
  'ResNet-101_64x64', 'ResNet-152_64x64', 'ResNet-200_64x64'
}
--]]
local data_name = {
  'ResNet-200_224x224', 
}

local criterion = nn.ClassNLLCriterion()

local nb_classes = 10 
for idx_exp = 1, #data_name do
  opt.exp_nb = idx_exp
  trainset, testset = load_dataset(data_name[opt.exp_nb])

  -- Defining classification model 
  C_model = nn.Sequential(); 
  C_model:add(nn.Linear(testset.data:size(2), 1024)):add(nn.ReLU())
  C_model:add(nn.Linear(1024, 512)):add(nn.ReLU())
  C_model:add(nn.Linear(512, 128)):add(nn.ReLU())
  C_model:add(nn.Linear(128, nb_classes)):add(nn.LogSoftMax())
  C_model = C_model:cuda()

  optimState = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
    beta2 = opt.beta2,
    epsilon = opt.epsilon
  }

  config = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
    beta2 = opt.beta2,
    epsilon = opt.epsilon
  }

  local input = torch.CudaTensor(opt.batchSize, testset.data:size(2))
  local label = torch.CudaTensor(opt.batchSize)
  local errM
  local epoch_tm = torch.Timer()
  local tm = torch.Timer()
  local data_tm = torch.Timer()
  local accuracies = torch.zeros(2, opt.niter)

  criterion = criterion:cuda()
  local p, gp = C_model:getParameters()
  p = p:normal(0, 1)
  local fx = function(x)
    if x ~= p then p:copy(x) end
    gp:zero()
    local output = C_model:forward(input)
    y = output:float()
    _, y_max = y:max(2)
    local errM = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    C_model:backward(input, df_do)
    confusion_train:batchAdd(y_max:squeeze():float(), label:float())
    return errM, gp
  end

  local function test_model()
    local confusion = optim.ConfusionMatrix(nb_classes)
    confusion:zero()
    for idx = 1, opt.testSize, opt.batchSize do
      --xlua.progress(idx, opt.testSize)
      indices = torch.range(idx, math.min(idx+opt.batchSize, opt.testSize))
      local batch = getBatch(testset, indices)
      local y = C_model:forward(batch.data)
      y = y:float()
      _, y_max = y:max(2)
      confusion:batchAdd(y_max:squeeze():float(), batch.labels:float())
    end
    confusion:updateValids()
    return confusion
  end

  C_model:training()

  firstEpochAccuracies = torch.zeros(3*math.floor(opt.trainSize/opt.batchSize))
  idx = 1
  for epoch = 1, opt.niter do
    local indices_rand = torch.randperm(opt.trainSize)
    confusion_train = optim.ConfusionMatrix(nb_classes)
    confusion_train:zero()
    for i = 1, math.floor(opt.trainSize/opt.batchSize) do 
      xlua.progress(i, math.floor(opt.trainSize/opt.batchSize))
      indices = indices_rand[{{1+(i-1)*opt.batchSize, i*opt.batchSize}}]
      batch =  getBatch(trainset, indices)
      input:copy(batch.data)
      label:copy(batch.labels)
      optim.adam(fx, p, config, optimState)
      C_model:clearState()
      p, gp = C_model:getParameters()
      if epoch <=3 and i%10 == 0 then
	local conf = test_model()
	firstEpochAccuracies[idx] = conf.totalValid; idx = idx + 1	
	print('First epochs accuracies: '); print(conf.totalValid)
      end
    end
    confusion_train:updateValids()
    C_model:evaluate()
    local conf = test_model()
    accuracies[1][epoch] = confusion_train.totalValid
    accuracies[2][epoch] = conf.totalValid
    C_model:training()
    print('Epoch: ' .. epoch .. ' out of ' .. opt.niter  .. '; Test performance: ' .. accuracies[2][epoch] .. '; Train performance: ' .. accuracies[1][epoch])
  end
  torch.save('first_epochs_' .. data_name[opt.exp_nb] .. '.t7', firstEpochAccuracies)
--  torch.save('accuracies_' .. data_name[opt.exp_nb] .. '.t7', accuracies)
end
