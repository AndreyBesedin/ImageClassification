-- run with DATA_ROOT=/home/besedin/workspace/Data/LSUN/ls
require 'cunn'
require 'cudnn'
require 'image'
require 'optim'
require 'nngraph'

-- Advanced options for training
opt = {
  lr = 0.001,
  initClassNb = 4, -- Number of already pretrained classes in the model
  pretrainedClasses = { 2, 3, 4, 5},
  maxClassNb = 5,      -- Maximum nb of classes in any stream interval
  usePretrainedModels = true,
  imSize = 224,
  miniBatchSize = 64,
  bufferSize = 100, -- Number of batches in the buffer
  gpu = 1,
  dropout = 0,
  testing = 'real',
  continue_training = false,
  totalClasses = 10, -- Total nb of classes in stream, basically unknown but since we use static datasets as stream, let's say we know it... 
}

local data_classes = {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 
                      'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'}
opt.full_data_classes = data_classes
opt.manualSeed = torch.random(1, 10000) -- fix seed
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(1)

-------------------------------------------------------------------------------------------------------
-- FUNCTIONS TO INITIALIZE/LOAD MODELS
-------------------------------------------------------------------------------------------------------

-- LSUN_CLASSIFIER
local function init_classifier_LSUN(inSize, nbClasses, opt)
   -- Defining classification model 
  if opt.continue_training then
    local C = torch.load('./models/progress/LSUN_classifier.t7')
    return C
  end
  local C = nn.Sequential(); 
  C:add(nn.Linear(inSize, 1024)):add(nn.ReLU())
  C:add(nn.Linear(1024, 512)):add(nn.ReLU())
  C:add(nn.Linear(512, 128)):add(nn.ReLU())
  C:add(nn.Linear(128, nbClasses)):add(nn.LogSoftMax())
  if opt.gpu == 1 then C = C:cuda() end
  return C
end

-- FEATURE EXTRACTOR FOR LSUN IMAGES
local feature_extractor = torch.load('./models/feature_extractors/resnet-200.t7')
feature_extractor:remove(14); feature_extractor:remove(13)

-- DCGAN GENERATORS AND DISCRIMINATORS
local function load_pretrained_generators_LSUN(opt)
  local G = {}; local D = {}
  -- Initialize all the models with some pretrained model (let's say bridge generator)
  for idx_model = 1, #opt.full_data_classes do
    G[idx_model] = torch.load('./models/LSUN_generators/pretrained/init_G.t7')
    D[idx_model] = torch.load('./models/LSUN_generators/pretrained/init_D.t7')
  end
  -- Replace chosen classes with respective pretrained models
  for idx_model = 1, #opt.pretrainedClasses do
    G[idx_model] = torch.load('./models/LSUN_generators/pretrained/' .. opt.full_data_classes[opt.pretrainedClasses[idx_model]] .. '_G.t7')
    D[idx_model] = torch.load('./models/LSUN_generators/pretrained/' .. opt.full_data_classes[opt.pretrainedClasses[idx_model]] .. '_D.t7')
  end
  -- If we want to continue training from last checkpoints
  if opt.continue_training then
    print('Loading models from previous training')
    G[idx_model] = torch.load('./models/progress/LSUN_generators/' .. opt.full_data_classes[opt.pretrainedClasses[idx_model]] .. '_G.t7')
    D[idx_model] = torch.load('./models/progress/LSUN_generators' .. opt.full_data_classes[opt.pretrainedClasses[idx_model]] .. '_D.t7')
  end
  return G, D
end

-------------------------------------------------------------------------------------------------------
-- ACCESSING AND LOADING DATA
-------------------------------------------------------------------------------------------------------

local function initialize_loaders(opt)
  local data = {}; local N = #opt.full_data_classes
  for idx_class = 1, N do
    opt.data_classes = {opt.full_data_classes[idx_class]}
    DataLoader = dofile('./data.lua')
    data[idx_class] = DataLoader.new(opt.nThreads, 'lsun', opt)
  end
  return data
end

-- PRELOADING TESTSET, THIS ONE WON'T BE CHANGING 
local testset = torch.load(path_to_testset) -- TO DO


local criterion = nn.ClassNLLCriterion()
local nb_classes = 10 

-------------------------------------------------------------------------------------------------------
-- LOADING AND/OR PREDEFINING CLASSIFICATION AND GENERATIVE MODELS
------------------------------------------------------------------------------------------------------
GAN = {}
if opt.usePretrainedModels == true then
  -- Use pretrained classifier and generators for several classes
  C_model = torch.load('./pretrained_models/' .. dataset .. '/classifier.t7')
  for idx = 1, 10 do
    GAN[idx] = {}
    if idx <= opt.initClassNb then
      GAN[idx].D = torch.load('./pretrained_models/' .. dataset .. '/' .. data_classes[idx] .. '_D.t7')
      GAN[idx].G = torch.load('./pretrained_models/' .. dataset .. '/' .. data_classes[idx] .. '_G.t7')
    else 
      GAN[idx].D = init_D()
      GAN[idx].G = init_G()
    end
  end
else
  C_model = init_classifier(inSize, nbClasses, opt)
  for idx = 1, 10 do
    GAN[idx] = {}
    GAN[idx].D = init_D()
    GAN[idx].G = init_G()
  end
end

---------------------------------------------------------------------------------------------------------
-- FUNCTION TO FORM THE STREAM 
---------------------------------------------------------------------------------------------------------
local function get_new_interval(old_classes, opt)
  --[[ 
  - In this function we define our stream. We start it with N classes, and at each new interval of the stream we make some classes
  dissappear and some others appear in the stream. 
  - The intervals are of random length, that is taken uniformly from 500 to 1000 batches per interval.
  - By the end of each interval we remove several classes and add some new so that the number of classes never exceeds OPT.MAXCLASSNB, and 
  in every new interval there is at least 1 and at most N-1 classes from previous interval, where N is the nb of classes in previous interval.
-- ]]
  local interval_classes = get_new_classes(old_classes, opt)
  -- Defining the interval
  local interval_length = math.floor(torch.uniform(500,1000))
  local interval = torch.zeros(interval_length)
  local idx_batch = 1
  if not current_class then current_class = 1 end
  while idx_batch < interval_length do
    -- Choose class from available in the interval, but different from the current
    current_class = interval_classes[interval_classes:ne(current_class)][torch.random(interval_classes:size(1)-1)] 
    -- Number of consequent batches that come from one class
    local class_duration = math.floor(torch.uniform(5,30))
    interval[{{idx_batch, math.min(idx_batch + class_duration, interval_length)}}]:fill(current_class);
    idx_batch = idx_batch + class_duration
  end
  return interval, interval_classes
end
  
local function get_new_classes(classes, opt)
  local available_classes = torch.range(1, opt.totalClasses)
  -- We have limited nb of classes in one interval
  maxClassNb = opt.maxClassNb
  -- Nb of classes in previous stream interval:
  local N = classes:size(1)                                                    
  -- Keeping at least one and at max N-1 classes:
  local nb_of_classes_to_keep = math.floor(torch.uniform(1, N - 0.0001))     
  --Choose the classes we keep in the strezam from previous interval
  local classes_to_keep = torch.randperm(N)[{{1, nb_of_classes_to_keep}}]:long()
  classes = classes:index(1, classes_to_keep)
  N = classes:size(1)
  -- Don't want to add classes, that are already in the stream
  for idx = 1, N do 
    available_classes = available_classes[available_classes:ne(classes[idx])]
  end
  -- Decide how many classes we add: 
  local nb_of_classes_to_add = math.floor(torch.uniform(1, maxClassNb - N + 0.999))
  -- And choosing those classes from the available ones: 
  local classes_to_add_id = torch.randperm(available_classes:size(1))[{{1, nb_of_classes_to_add}}]:long()
  local classes_to_add = available_classes:index(1, classes_to_add_id)
  -- Completing the classes list
  for idx = 1, classes_to_add:size(1) do
    classes = classes:cat(torch.Tensor({classes_to_add[idx]}))
  end
  return classes
end

---------------------------------------------------------------------------------------------------------
-- FUNCTION TO MANIPULATE THE BUFFER 
---------------------------------------------------------------------------------------------------------
local function init_buffer(opt)
  local buffer = torch.zeros(10, opt.bufferSize*opt.batchSize, 2048)
  local buffer_count = torch.zeros(10)
  return buffer, buffer_count
end

local function complete_buffer(buffer, buffer_count, GAN, opt)
  local class_size = opt.bufferSize*opt.batchSize 
  local res = {}; res.data = torch.zeros(10*class_size, 2048); res.labels = torch.zeros(10*class_size)
  for idx_class = 1, 10 do
    for idx_gen = buffer_count[idx_class] + 1, opt.bufferSize do
      local gen_batch = generate_data(GAN[idx_class].G, opt.batchSize)
      gen_batch = rescale_batch(gen_batch, 224)
      buffer[{{idx_class},{1 + (idx_gen-1)*opt.batchSize, idx_gen*opt.batchSize},{}}] = feature_extractor:forward(gen_batch) 
    end
    res.labels[{{1 + (idx_class-1)*class_size, idx_class*class_size}}]:fill(idx_class)
    res.data[{{1  + buffer_count[idx_class]},{}}] = buffer[{{idx_class},{},{}}]:squeeze()
  end
  return res
end
---------------------------------------------------------------------------------------------------------
local old_classes = torch.DoubleTensor(opt.pretrainedClasses);  -- Initializing classes to the start of the stream
---------------------------------------------------------------------------------------------------------
-- INITIALIZING TRAINING AND PARAMETERS
---------------------------------------------------------------------------------------------------------
local data = initialize_loaders(opt)
local interval_is_over = true
local GAN_count = torch.zeros(10); local buffer_count = torch.zeros(10)
local buffer = torch.zeros(10, opt.bufferSize*opt.batchSize, 2048)
local Stream = true

local buffer, buffer_count = init_buffer(opt)

---------------------------------------------------------------------------------------------------------
-- TRAINING
---------------------------------------------------------------------------------------------------------
local function train_GAN(GAN_, data_)

end

while Stream do
  if interval_is_over == true then 
    classes = get_new_classes(old_classes) -- getting classes that would appear in the new interval
    interval = get_new_interval(old_classes) -- fill in the interval with ordered classes of batches from stream
    batch_idx = 1
    interval_is_over = false
  end  

  local current_class = interval[batch_idx]
  local batch_orig = data[current_class]:getBatch() 
  train_GAN(GAN[current_class], data_)
  local batch_features = feature_extractor:forward(batch_orig)
  
  buffer_count[current_class] = buffer_count[current_class] + 1
  GAN_count[current_class] = GAN_count[current_class] + 1
  buffer[{{current_class},{1 + (buffer_count[current_class]-1)*opt.batchSize, buffer_count[current_class]*opt.batchSize},{}}] = batch_features
  if buffer_count[current_class] == opt.bufferSize then
    buffer = complete_buffer(buffer, GAN)
    C_model = train_classifier(C_model, buffer)
    test_classifier(C_model, testset)
    buffer, buffer_count = init_buffer(opt)
  end

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

  --firstEpochAccuracies = torch.zeros(3*math.floor(opt.trainSize/opt.batchSize))
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
      --if epoch <=3 and i%100 == 0 then
      --  local conf = test_model()
        --firstEpochAccuracies[idx] = conf.totalValid; idx = idx + 1  
      --  print('First epochs accuracies: '); print(conf.totalValid)
      --end
    end
    local conf = test_model()
    print('test_accuracy: ')
    print(conf)
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
