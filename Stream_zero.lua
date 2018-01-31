-------------------------------------------------------------------------------------------------------
-- DESCRIPTION
-- should run with DATA_ROOT=/home/besedin/workspace/Data/LSUN/...
-------------------------------------------------------------------------------------------------------

-------------------------------------------------------------------------------------------------------
-- LOADING PACKAGES
-------------------------------------------------------------------------------------------------------
print('LOADING DEPENDENCIES...')
require 'cunn'
require 'cudnn'
require 'image'
require 'optim'
require 'nngraph'
dofile('./tools/tools.lua')
-------------------------------------------------------------------------------------------------------
-- ADVANCED OPTIONS FOR TRAINING
-------------------------------------------------------------------------------------------------------
opt = {
  lr = 0.001,
  nThreads = 6,
  initClassNb = 4, -- Number of already pretrained classes in the model
  pretrainedClasses = {2, 3, 4, 5},
  maxClassNb = 5,      -- Maximum nb of classes in any stream interval
  usePretrainedModels = true,
  imSize = 224,
  batchSize = 16,
  loadSize = 256,
  fineSize = 224,
  interval_size = {300, 400},
  bufferSize = 50, -- Number of batches in the buffer
  gpu = 1,
  dropout = 0,
  testing = 'real',
  continue_training = false,
  init_pretrained = false,
  totalClasses = 10, -- Total nb of classes in stream, basically unknown but since we use static datasets as stream, let's say we know it... 
}

-- Full list of available classes
local data_classes = {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 
                      'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'}                      
-------------------------------------------------------------------------------------------------------
-- MODIFYING OPTIONS
-------------------------------------------------------------------------------------------------------
opt.nb_classes = #data_classes                      
opt.full_data_classes = data_classes
opt.manualSeed = torch.random(1, 10000) -- fix seed

-------------------------------------------------------------------------------------------------------
-- SETTING DEFAULT TORCH BEHAVIOR
-------------------------------------------------------------------------------------------------------
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(1)

-------------------------------------------------------------------------------------------------------
-- FUNCTIONS TO INITIALIZE/LOAD MODELS
-------------------------------------------------------------------------------------------------------

-- LSUN_CLASSIFIER
function init_classifier_LSUN(inSize, nbClasses, opt)
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
function init_feature_extractor(path_to_fs)
  local feature_extractor = torch.load('./models/feature_extractors/resnet-200.t7')
  feature_extractor:remove(14); 
  feature_extractor:remove(13); 
  feature_extractor:add(nn.View(2048)); 
  feature_extractor = feature_extractor:float()
  return feature_extractor:cuda()
end

-- DCGAN GENERATORS AND DISCRIMINATORS
function load_pretrained_generators_LSUN(opt)
  local GAN = {} 
  -- Initialize all the models with some pretrained model (let's say bridge generator)
  for idx_model = 1, #opt.full_data_classes do
    GAN[idx_model] = {}
    if opt.continue_training then
      GAN[idx_model].G = torch.load('./models/progress/LSUN_generators/' .. opt.full_data_classes[opt.pretrainedClasses[idx_model]] .. '_G.t7')
      GAN[idx_model].D = torch.load('./models/progress/LSUN_generators/' .. opt.full_data_classes[opt.pretrainedClasses[idx_model]] .. '_D.t7')
    elseif opt.init_pretrained == true then
      GAN[idx_model].G = torch.load('./models/LSUN_generators/pretrained/init_G.t7')
      GAN[idx_model].D = torch.load('./models/LSUN_generators/pretrained/init_D.t7')
    else
      GAN[idx_model].G = init_G()
      GAN[idx_model].D = init_D()
    end
  end
  if opt.continue_training then return GAN end
  -- Replace chosen classes with respective pretrained models
  for idx_model = 1, #opt.pretrainedClasses do
    GAN[idx_model].G = torch.load('./models/LSUN_generators/pretrained/' .. opt.full_data_classes[opt.pretrainedClasses[idx_model]] .. '_G.t7')
    GAN[idx_model].D = torch.load('./models/LSUN_generators/pretrained/' .. opt.full_data_classes[opt.pretrainedClasses[idx_model]] .. '_D.t7')
  end
  return GAN
end

function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

function init_G()
  local G = nn.Sequential()
  local ngf = 64; local nc = 3
  G:add(nn.SpatialFullConvolution(100, ngf * 8, 4, 4))
  G:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
  -- state size: (ngf*8) x 4 x 4
  G:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
  G:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
  -- state size: (ngf*4) x 8 x 8
  G:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
  G:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
  -- state size: (ngf*2) x 16 x 16
  G:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
  G:add(nn.SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
  -- state size: (ngf) x 32 x 32
  G:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
  G:add(nn.Tanh())
  G:apply(weights_init)
  return G
end

function init_D()
  local D = nn.Sequential()
  local nc = 3; local ndf = 64
  D:add(nn.SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
  D:add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf) x 32 x 32
  D:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
  D:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf*2) x 16 x 16
  D:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
  D:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf*4) x 8 x 8
  D:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
  D:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf*8) x 4 x 4
  D:add(nn.SpatialConvolution(ndf * 8, 1, 4, 4))
  D:add(nn.Sigmoid())
  -- state size: 1 x 1 x 1
  D:add(nn.View(1):setNumInputDims(3))
  D:apply(weights_init)
  return D
end
-------------------------------------------------------------------------------------------------------
-- ACCESSING, GENERATING AND LOADING DATA
-------------------------------------------------------------------------------------------------------
function initialize_loaders(opt)
  print('\nINITIALIZING DATA LOADERS');
  local data = {}; local N = #opt.full_data_classes
  for idx_class = 1, N do
    print('Reading ' .. opt.full_data_classes[idx_class] .. ' DB')
    opt.data_classes = {opt.full_data_classes[idx_class]}
    DataLoader = dofile('./data/data.lua')
    data[idx_class] = DataLoader.new(opt.nThreads, 'lsun', opt)
  end
  return data
end

function generate_data(G, batchSize)
  local noise = torch.FloatTensor(batchSize, 100, 1, 1):cuda()
  local batch = G:forward(noise:normal(0,1))
  batch = batch:float()
  return batch:cuda()
end

 function rescale_3D_batch(batch, outSize)
  if #batch:size()<4 then error('not 3D data batch') end
  if batch:size(3)~=batch:size(4) then error('images are not square') end
  local batchSize = batch:size(1)
  local imSize = batch:size(3)
  local new_batch = torch.FloatTensor(batchSize, 3, outSize, outSize)
  for idx_im = 1, batchSize do
    new_batch[idx_im] = image.scale(batch[idx_im], outSize)
  end
  return new_batch:float()
end
---------------------------------------------------------------------------------------------------------
-- FUNCTION TO FORM THE STREAM 
---------------------------------------------------------------------------------------------------------
function get_new_interval(classes, opt)
  --[[ 
  - In this function we define our stream. We start it with N classes, and at each new interval of the stream we make some classes
  dissappear and some others appear in the stream. 
  - The intervals are of random length, that is taken uniformly from 500 to 1000 batches per interval.
  - By the end of each interval we remove several classes and add some new so that the number of classes never exceeds OPT.MAXCLASSNB, and 
  in every new interval there is at least 1 and at most N-1 classes from previous interval, where N is the nb of classes in previous interval.
-- ]]
  local interval_classes = get_new_classes(classes, opt)
  -- Defining the interval
  local interval_length = math.floor(torch.uniform(opt.interval_size[1],opt.interval_size[2]))
  --print('New interval contains ' .. interval_length .. ' batches')
  local interval = torch.zeros(interval_length)
  local idx_batch = 1
  if not current_class then current_class = 1 end
  while idx_batch < interval_length do
    -- Choose class from available in the interval, but different from the current
    current_class = interval_classes[interval_classes:ne(current_class)][torch.random(interval_classes:size(1)-1)] 
    -- Number of consequent batches that come from one class
    local class_duration = math.floor(torch.uniform(5,15))
    --print('Class: ' .. current_class .. ', duration: ' .. class_duration .. ' batches')
    interval[{{idx_batch, math.min(idx_batch + class_duration, interval_length)}}]:fill(current_class);
    idx_batch = idx_batch + class_duration
  end
  return interval, interval_classes
end
  
function get_new_classes(classes, opt)
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
function init_buffer(opt)
  local buffer = torch.zeros(10, opt.bufferSize*opt.batchSize, 2048)
  local buffer_count = torch.zeros(10)
  print('\nBUFFER (RE)INITIALIZED, BATCH COUNT SET TO ZERO')
  return buffer, buffer_count
end

function complete_buffer(buffer, buffer_count, GAN, feature_extractor, opt)
  --[[
  Function to complete the buffer with generated data.
  ]]
  print('\nCOMPLETENIG BUFFER WITH GENERATED DATA');
  local class_size = opt.bufferSize * opt.batchSize 
  local res = {}; res.data = torch.zeros(10*class_size, 2048); res.labels = torch.zeros(10*class_size)
  for idx_class = 1, 10 do
    print('Generating ' .. opt.full_data_classes[idx_class] .. 's')
    for idx_gen = buffer_count[idx_class] + 1, opt.bufferSize do
      xlua.progress(idx_gen, opt.bufferSize)
      local gen_batch_small = generate_data(GAN[idx_class].G, opt.batchSize)
      local gen_batch_big = rescale_3D_batch(gen_batch_small:float(), 224)
      local features = feature_extractor:forward(gen_batch_big:cuda()) 
      buffer[{{idx_class},{1 + (idx_gen-1)*opt.batchSize, idx_gen*opt.batchSize},{}}] = features:float()
    end
    res.labels[{{1 + (idx_class-1)*class_size, idx_class*class_size}}]:fill(idx_class)
    res.data[{{1  + (idx_class-1)*class_size, idx_class*class_size},{}}] = buffer[{{idx_class},{},{}}]:squeeze()
  end
  return res
end

---------------------------------------------------------------------------------------------------------
-- TRAINING/TESTING FUNCTIONS 
---------------------------------------------------------------------------------------------------------

function train_GAN(GAN, data, optimState_)  
  parametersD, gradParametersD = GAN.D:getParameters()
  parametersG, gradParametersG = GAN.G:getParameters()
  input = torch.CudaTensor(opt.batchSize, 3, 64, 64)
  noise = torch.CudaTensor(opt.batchSize, 100, 1, 1)
  label = torch.CudaTensor(opt.batchSize)
  local fDx = function(x)
    gradParametersD:zero()
    -- train with real
    local real = data:cuda()
    input:copy(real)
    label:fill(1)
    local output = GAN.D:forward(input)
    local errD_real = GAN_criterion:forward(output, label)
    local df_do = GAN_criterion:backward(output, label)
    GAN.D:backward(input, df_do)

    -- train with fake
    noise:normal(0, 1)
    local fake = GAN.G:forward(noise)
    input:copy(fake)
    label:fill(0)

    local output = GAN.D:forward(input)
    local errD_fake = GAN_criterion:forward(output, label)
    local df_do = GAN_criterion:backward(output, label)
    GAN.D:backward(input, df_do)

    errD = errD_real + errD_fake
    return errD, gradParametersD
  end
  
  local fGx = function(x)
    gradParametersG:zero()

    --[[ the three lines below were already executed in fDx, so save computation
    noise:uniform(-1, 1) -- regenerate random noise
    local fake = netG:forward(noise)
    input:copy(fake) ]]--
    label:fill(1) -- fake labels are real for generator cost

    local output = GAN.D.output -- netD:forward(input) was already executed in fDx, so save computation
    errG = GAN_criterion:forward(output, label)
    local df_do = GAN_criterion:backward(output, label)
    local df_dg = GAN.D:updateGradInput(input, df_do)

    GAN.G:backward(noise, df_dg)
    return errG, gradParametersG
  end

  optim.adam(fDx, parametersD, optimState_.D)
  optim.adam(fGx, parametersG, optimState_.G)
  parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
  parametersG, gradParametersG = nil, nil
  collectgarbage()
  return GAN, errD, errG
end

function train_classifier(C_model, data, opt)
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
  
  input = torch.CudaTensor(opt.batchSize, data.data:size(2))
  label = torch.CudaTensor(opt.batchSize)
  criterion = classif_criterion:cuda()
  p, gp = C_model:getParameters()
  p = p:normal(0, 1)
  local indices_rand = torch.randperm(data.data:size(1))
  confusion_train = optim.ConfusionMatrix(10)
  confusion_train:zero()
  
  for i = 1, math.floor(data.data:size(1)/opt.batchSize) do 
    xlua.progress(i, math.floor(data.data:size(1)/opt.batchSize))
    indices = indices_rand[{{1+(i-1)*opt.batchSize, i*opt.batchSize}}]
    batch = getBatch(data, indices:long())
    input:copy(batch.data)
    label:copy(batch.labels)
    optim.adam(fx, p, config, optimState)
    C_model:clearState()
    p, gp = C_model:getParameters()
  end
  print('Training set confusion matrix: ')
  print(confusion_train)
  return C_model
end

function getBatch(data, indices)
  local batch = {}
  batch.data = data.data:index(1, indices:long())
  batch.labels = data.labels:index(1, indices:long())
  return batch
end

function test_classifier(C_model, data)
  local confusion = optim.ConfusionMatrix(10)
  confusion:zero()
  for idx = 1, data.data:size(1), opt.batchSize do
    --xlua.progress(idx, opt.testSize)
    indices = torch.range(idx, math.min(idx + opt.batchSize, data.data:size(1)))
    local batch = getBatch(testset, indices:long())
    local y = C_model:forward(batch.data:cuda())
    y = y:float()
    _, y_max = y:max(2)
    confusion:batchAdd(y_max:squeeze():float(), batch.labels:float())
  end
  confusion:updateValids()
  return confusion  
end

---------------------------------------------------------------------------------------------------------
-- INITIALIZING TRAINING AND PARAMETERS
---------------------------------------------------------------------------------------------------------
local classes = torch.FloatTensor(opt.pretrainedClasses);  -- Initializing classes to the start of the stream

classif_criterion = nn.ClassNLLCriterion()
GAN_criterion = nn.BCECriterion()
GAN_criterion = GAN_criterion:cuda()

local test_res = {}

optimState = {
  learningRate = opt.lr,
}

config = {
  learningRate = opt.lr,
}

---------------------------------------------------------------------------------------------------------
-- PRELOADING TESTSET, THIS ONE WON'T BE CHANGING 
---------------------------------------------------------------------------------------------------------

print('\nLOADING THE TESTSET')
path_to_testset = './subsets/full/testset_5k_per_class_1.t7'
testset = torch.load(path_to_testset)
print('\nTESTSET LOADED, SIZE: ' .. testset.data:size(1)); 

---------------------------------------------------------------------------------------------------------
-- INITIALIZING MODELS
---------------------------------------------------------------------------------------------------------

print('\nINITIALIZING CLASSIFICATION MODEL'); C_model = init_classifier_LSUN(2048, 10, opt)
print('\nINITIALIZING DCGANs'); GAN = load_pretrained_generators_LSUN(opt)
print('\nLOADING FEATURE EXTRACTOR'); feature_extractor = init_feature_extractor('./models/feature_extractors/resnet-200.t7')
print('\nMODELS INITIALIZED, START TRAINING'); sleep(2)

p, gp = C_model:getParameters()
---------------------------------------------------------------------------------------------------------
-- TRAINING
---------------------------------------------------------------------------------------------------------

-- TEST for buffer completion
--buffer_count[1] = 55; buffer_count[2] = 10;  buffer_count[3] = opt.bufferSize;
--res = complete_buffer(buffer, buffer_count, GAN, feature_extractor, opt)
--print('Test complete, please check res')

to_save = {}
buffer, buffer_count = init_buffer(opt)
idx_buffer = 0
while Stream do
  collectgarbage() 
  buffer = complete_buffer(buffer, buffer_count, GAN, feature_extractor, opt)
  C_model = train_classifier(C_model, buffer, opt)
  idx_buffer = idx_buffer + 1
  buffer, buffer_count = init_buffer(opt)
  if batch_idx_buffer % 10 == then 
    confusion = test_classifier(C_model, testset); print(confusion)
    to_save[idx_buffer] = confusion
    torch.save('./results/LSUN/stream/confusions_zero.t7', to_save)
  end
end
