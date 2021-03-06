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
  imSize = 28,
  batchSize = 16,
  interval_size = {300, 400},
  bufferSize = 50, -- Number of batches in the buffer
  gpu = 1,
  dropout = 0,
  epoch_nb = 2,
  testing = 'real',
  continue_training = false,
  init_pretrained = false,
  totalClasses = 10, -- Total nb of classes in stream, basically unknown but since we use static datasets as stream, let's say we know it... 
  generatedOnly = true,
  train_batch_fake = false,
  train_batch_real = false,
}

-- Full list of available classes
data_classes = {'one', 'two', 'three', 'four', 
                      'five', 'six', 'seven', 'eight', 'nine', 'zero'}                      
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

function deepcopy(orig)
  local orig_type = type(orig)
  local copy
  if orig_type == 'table' then
    copy = {}
    for orig_key, orig_value in next, orig, nil do
      copy[deepcopy(orig_key)] = deepcopy(orig_value)
    end
    setmetatable(copy, deepcopy(getmetatable(orig)))
  else -- number, string, boolean, etc
    copy = orig
  end
return copy
end
-------------------------------------------------------------------------------------------------------
-- FUNCTIONS TO INITIALIZE/LOAD MODELS
-------------------------------------------------------------------------------------------------------

-- LSUN_CLASSIFIER
function init_classifier_MNIST(opt)
   -- Defining classification model 
  if opt.continue_training then
    local C = torch.load('./models/progress/MNIST_classifier.t7')
    return C
  end
  local C = nn.Sequential(); 
  C:add(nn.SpatialConvolution(1, 16, 4, 4, 2, 2, 1, 1))
  C:add(nn.ReLU(true))
  C:add(nn.SpatialConvolution(16, 32, 4, 4, 2, 2, 1, 1))
  C:add(nn.ReLU(true)):add(nn.SpatialMaxPooling(2,2,2,2):ceil())
  C:add(nn.View(512))
  C:add(nn.Linear(512, 512)):add(nn.Dropout(0.3)):add(nn.ReLU())
  C:add(nn.Linear(512, 128)):add(nn.ReLU())
  C:add(nn.Linear(128, 10)):add(nn.LogSoftMax())
  C = C:float()
  if opt.gpu == 1 then C = C:cuda() end
  return C
end


-- DCGAN GENERATORS AND DISCRIMINATORS

function load_pretrained_generators_MNIST(opt)
  local GAN = {} 
  -- Initialize all the models with some pretrained model (let's say bridge generator)
  for idx_model = 1, #opt.full_data_classes do
    GAN[idx_model] = {}
    if opt.continue_training then
      GAN[idx_model].G = torch.load('./models/progress/MNIST_generators/' .. opt.full_data_classes[opt.pretrainedClasses[idx_model]] .. '_G.t7')
      GAN[idx_model].D = torch.load('./models/progress/MNIST_generators/' .. opt.full_data_classes[opt.pretrainedClasses[idx_model]] .. '_D.t7')
    elseif opt.init_pretrained == true then
      GAN[idx_model].G = torch.load('./models/MNIST_generators/pretrained/one_G.t7')
      GAN[idx_model].D = torch.load('./models/MNIST_generators/pretrained/one_D.t7')
    else
      GAN[idx_model].G = init_G()
      GAN[idx_model].D = init_D()
    end
  end
  if opt.continue_training then return GAN end
  -- Replace chosen classes with respective pretrained models
  for idx_model = 1, #opt.pretrainedClasses do
    GAN[opt.pretrainedClasses[idx_model]].G = torch.load('./models/MNIST_generators/pretrained/' .. opt.full_data_classes[opt.pretrainedClasses[idx_model]] .. '_G.t7')
    GAN[opt.pretrainedClasses[idx_model]].D = torch.load('./models/MNIST_generators/pretrained/' .. opt.full_data_classes[opt.pretrainedClasses[idx_model]] .. '_D.t7')
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
  local ngf = 64; local nc = 1
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
  return G:cuda()
end

function init_D()
  local D = nn.Sequential()
  local nc = 1; local ndf = 64
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
  return D:cuda()
end
-------------------------------------------------------------------------------------------------------
-- ACCESSING, GENERATING AND LOADING DATA
-------------------------------------------------------------------------------------------------------

function generate_data(G, batchSize)
  local noise = torch.FloatTensor(batchSize, 100, 1, 1):cuda()
  local batch = G:forward(noise:normal(0,1))
--  batch = batch:float()
  return batch:cuda()
end

function rescale_3D_batch(batch, outSize)
  if #batch:size()<4 then error('not 3D data batch') end
  if batch:size(3)~=batch:size(4) then error('images are not square') end
  local batchSize = batch:size(1)
  local imSize = batch:size(3)
  local new_batch = torch.FloatTensor(batchSize, 1, outSize, outSize)
  for idx_im = 1, batchSize do
    new_batch[idx_im] = image.scale(batch[idx_im], outSize)
  end
  return new_batch:float()
end

function trainset_by_class(trainset)
  local indices_ = torch.ones(10):float()
  local res = {} 
  local res_ = {}
  for idx = 1, 10 do res[idx] = {}; end
  for idx = 1, trainset.data:size(1) do
    res[trainset.labels[idx]][indices_[trainset.labels[idx]]] = trainset.data[idx]
    indices_[trainset.labels[idx]] = indices_[trainset.labels[idx]] +1
  end
  for idx = 1, 10 do
    res_[idx] = torch.cat(res[idx],1):reshape(#res[idx], 1, 28, 28)
  end
  return res_
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
  local buffer = torch.zeros(10, opt.bufferSize*opt.batchSize, 1, 28, 28)
  local buffer_count = torch.zeros(10)
  print('\nBUFFER (RE)INITIALIZED, BATCH COUNT SET TO ZERO')
  return buffer, buffer_count
end

function complete_buffer(buffer, buffer_count, GAN, opt)
  --[[
  Function to complete the buffer with generated data.
  ]]
  print('\nCOMPLETENIG BUFFER WITH GENERATED DATA');
  local class_size = opt.bufferSize * opt.batchSize 
  local res = {}; res.data = torch.zeros(10*class_size, 1, 28,28); res.labels = torch.zeros(10*class_size)
  for idx_class = 1, 10 do
    print('Generating ' .. opt.full_data_classes[idx_class] .. 's')
    for idx_gen = buffer_count[idx_class] + 1, opt.bufferSize do
      xlua.progress(idx_gen, opt.bufferSize)
      local gen_batch_small = generate_data(GAN[idx_class].G, opt.batchSize)
      local gen_batch_big = rescale_3D_batch(gen_batch_small:float(), 28)
      buffer[{{idx_class},{1 + (idx_gen-1)*opt.batchSize, idx_gen*opt.batchSize},{},{},{}}] = gen_batch_big
    end
    res.labels[{{1 + (idx_class-1)*class_size, idx_class*class_size}}]:fill(idx_class)
    res.data[{{1  + (idx_class-1)*class_size, idx_class*class_size},{},{},{}}] = buffer[{{idx_class},{},{},{},{}}]
  end
  return res
end

---------------------------------------------------------------------------------------------------------
-- TRAINING/TESTING FUNCTIONS 
---------------------------------------------------------------------------------------------------------

function train_GAN(GAN, data, optimState_)  
  parametersD, gradParametersD = GAN.D:getParameters()
  parametersG, gradParametersG = GAN.G:getParameters()
  input = torch.CudaTensor(opt.batchSize, 1, 64, 64)
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
  C_model:training()
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
  input = torch.CudaTensor(opt.batchSize, 1, 28, 28)
  label = torch.CudaTensor(opt.batchSize)
  criterion = classif_criterion:cuda()
  local indices_rand = torch.randperm(data.data:size(1))
  confusion_train = optim.ConfusionMatrix(10)
  confusion_train:zero()
  idx_test = 0
  confusion_test_ = {}
  for i = 1, math.floor(data.data:size(1)/opt.batchSize) do 
    xlua.progress(i, math.floor(data.data:size(1)/opt.batchSize))
    indices_ = indices_rand[{{1+(i-1)*opt.batchSize, i*opt.batchSize}}]
    batch = getBatch(data, indices_:long())
    input:copy(batch.data)
    label:copy(batch.labels)
    optim.adam(fx, p, config, optimState)
    --C_model:clearState()
    p, gp = C_model:getParameters()
--    if i%450==0 then idx_test = idx_test + 1; confusion_test_[idx_test] = test_classifier(C_model, testset); print(confusion_test_[idx_test]) end
  end
  print('Training set confusion matrix: ')
  print(confusion_train)
  return C_model, confusion_test_
end

function getBatch(data, indices_)
  local batch = {}
  batch.data = data.data:index(1, indices_:long())
  batch.labels = data.labels:index(1, indices_:long())
  return batch
end

function getBatchFromTrainset(trainset_class, indices_)
  batch = trainset_class:index(1, indices_:long())
  return batch
end

function test_classifier(C_model, data)
  C_model:evaluate()  
  local confusion = optim.ConfusionMatrix(10)
  confusion:zero()
  for idx = 1, data.data:size(1), opt.batchSize do
    --xlua.progress(idx, opt.testSize)
    indices_ = torch.range(idx, math.min(idx + opt.batchSize, data.data:size(1)))
    local batch = getBatch(data, indices_:long())
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
        
 interval_is_over = true; interval_idx = 0
GAN_count = torch.zeros(10); buffer_count = torch.zeros(10)

Stream = true
classes = torch.FloatTensor(opt.pretrainedClasses);  -- Initializing classes to the start of the stream
-- Parameters for the training step zero:

--buffer, buffer_count = init_buffer(opt)

classif_criterion = nn.ClassNLLCriterion()
GAN_criterion = nn.BCECriterion()
GAN_criterion = GAN_criterion:cuda()

test_res = {}
optimState_GAN = {}
for idx = 1, 10 do
  optimState_GAN[idx]= {}
  optimState_GAN[idx].D = {learningRate=0.0002, beta1 = 0.5}
  optimState_GAN[idx].G = {learningRate=0.0002, beta1 = 0.5}
end

config = {
  learningRate = opt.lr,
  learningRateDecay = 1e-4,
  weightDecay = 1e-4,
}

---------------------------------------------------------------------------------------------------------
-- PRELOADING TESTSET, THIS ONE WON'T BE CHANGING 
---------------------------------------------------------------------------------------------------------

print('\nLOADING THE TESTSET')
path_to_testset = './datasets/MNIST/t7/testset.t7'
testset = torch.load(path_to_testset)
print('\nTESTSET LOADED, SIZE: ' .. testset.data:size(1)); 
path_to_trainset = './datasets/MNIST/t7/trainset.t7'
trainset = torch.load(path_to_trainset)
trainset_class = trainset_by_class(trainset) 
print('\nTRAINSET LOADED, SIZE: ' .. trainset.data:size(1));
---------------------------------------------------------------------------------------------------------
-- INITIALIZING MODELS
---------------------------------------------------------------------------------------------------------

print('\nINITIALIZING CLASSIFICATION MODEL'); C_model = init_classifier_MNIST(opt)
print('\nINITIALIZING DCGANs'); GAN = load_pretrained_generators_MNIST(opt)
print('\nMODELS INITIALIZED, START TRAINING');

p, gp = C_model:getParameters()
--p = p:normal(0, 1)
---------------------------------------------------------------------------------------------------------
-- TRAINING
---------------------------------------------------------------------------------------------------------

-- TEST for buffer completion
--buffer_count[1] = 55; buffer_count[2] = 10;  buffer_count[3] = opt.bufferSize;
--res = complete_buffer(buffer, buffer_count, GAN, feature_extractor, opt)
--print('Test complete, please check res')

-- Step zero: Testing on just pretrained GANs for initial classes:
if opt.train_batch_fake then
  opt_zero = deepcopy(opt); opt_zero.bufferSize = opt.bufferSize*10;
  --buffer_zero, buffer_count_zero = init_buffer(opt_zero)
  --buffer_zero = complete_buffer(buffer_zero, buffer_count_zero, GAN, opt_zero)
  confusion_test = {}
  confusion_test[0] = test_classifier(C_model, testset); print(confusion_test[0])
  for epoch = 1, opt.epoch_nb do
    buffer_zero, buffer_count_zero = init_buffer(opt_zero)
    buffer_zero = complete_buffer(buffer_zero, buffer_count_zero, GAN, opt_zero)
    confusion_test[epoch] = {}
    for idx = 1, 1 do
      C_model, confusion = train_classifier(C_model, buffer_zero, opt_zero)
      confusion_test[epoch] = {confusion}
      --confusion = test_classifier(C_model, testset); print(confusion)
    end
    torch.save('results/batch_training/MNIST_fake.t7', confusion_test)
  --  print('optim state: '); print(optimState) 
  end
end

if opt.train_batch_real then
  confusion_test = {}
  confusion_test[0] = test_classifier(C_model, testset); print(confusion_test[0])
  for epoch = 1, opt.epoch_nb do
    for idx = 1, 1 do
      C_model, confusion = train_classifier(C_model, trainset, opt)
      confusion_test[epoch] = confusion
      --confusion = test_classifier(C_model, testset); print(confusion)
    end
    torch.save('results/batch_training/MNIST_real.t7', confusion_test)
  --  print('optim state: '); print(optimState) 
  end
end

function generate_image_grid(GAN, visu_noise)
  local im_full = torch.FloatTensor(640,640):cuda()
  for idx_class = 1, 10 do
    gen_images = GAN[idx_class].G:forward(visu_noise:cuda())
    for idx_im = 1, 10 do
      im_full[{{1+64*(idx_im-1), 64*idx_im},{1+64*(idx_class - 1), 64*idx_class}}]=gen_images[idx_im]:squeeze():float()
    end
  end
  return (im_full+1)/2
end
to_save = {}
to_save.confusion = {}
to_save.GAN_count = {}
to_save.intervals = {}
to_save.intervals.duration = {}
to_save.intervals.classes = {}
to_save.confusion[0] = test_classifier(C_model, testset); print(to_save.confusion[0])
to_save.GAN_count[0] = torch.zeros(10)
to_save.intervals.duration[0] = 0
to_save.intervals.classes[0] = classes
buffer, buffer_count = init_buffer(opt)
visu_noise = torch.FloatTensor(10, 100, 1, 1); visu_noise = visu_noise:normal(0,1); visu_noise =visu_noise:cuda()

indices = {}
for idx = 1, 10 do indices[idx]= torch.randperm(trainset_class[idx]:size(1)):float() end
while Stream do
  collectgarbage()
  if interval_is_over == true then 
    print('\nInterval ' .. interval_idx .. ' is over, starting next')
    interval_idx = interval_idx + 1
--    classes = get_new_classes(classes, opt) -- getting classes that would appear in the new interval
    interval, classes = get_new_interval(classes, opt) -- fill in the interval with ordered classes of batches from stream
    to_save.intervals.duration[interval_idx] = interval:size(1)
    to_save.intervals.classes[interval_idx] = classes
    print('New classes: '); print(classes:reshape(1,classes:size(1)))
    batch_idx = 1
    interval_is_over = false
  end 
  local current_class = interval[batch_idx]
  batch_idx = batch_idx + 1
  if indices[current_class]:size(1) <= opt.batchSize then
    indices[current_class] = torch.randperm(trainset_class[current_class]:size(1))
  end
  indices_batch = indices[current_class][{{1,opt.batchSize}}]
  indices[current_class] = indices[current_class][{{opt.batchSize+1, indices[current_class]:size(1)}}]
  --print(indices[current_class]:size())
  batch_orig = trainset_class[current_class]:index(1, indices_batch:long())
  batch = rescale_3D_batch(batch_orig:float(), 64)
  --print('RECEIVED DATA FROM CLASS ' .. current_class)
  GAN[current_class], errD, errG = train_GAN(GAN[current_class], batch, optimState_GAN[current_class])
  --print('Class ' .. current_class .. ', errD = ' .. errD .. ', errG = ' .. errG)
  -- Filling in the buffer
  buffer_count[current_class] = buffer_count[current_class] + 1
  xlua.progress(buffer_count:sum(), opt.bufferSize*classes:size(1))
  GAN_count[current_class] = GAN_count[current_class] + 1
  buffer[{{current_class},{1 + (buffer_count[current_class]-1)*opt.batchSize, buffer_count[current_class]*opt.batchSize},{},{},{}}] = batch_orig:float()
  if buffer_count[current_class] == opt.bufferSize then
    print('Collected enough data. Samples distribution by class: '); print(buffer_count:reshape(1,10)) 
    buffer = complete_buffer(buffer, buffer_count, GAN, opt)
    print('Training clasifier with collected data')
    for epoch = 1, opt.epoch_nb do
      C_model = train_classifier(C_model, buffer, opt)
    end
    buffer, buffer_count = init_buffer(opt)
  end
  if batch_idx == interval:size(1) then 
    interval_is_over = true
    print('Currently real images fed to GANS, per class: '); print(GAN_count:reshape(1, 10)*opt.batchSize)
    confusion = test_classifier(C_model, testset); print(confusion)
    im_to_save = generate_image_grid(GAN, visu_noise)
    image.save('./results/MNIST/image_grids/interval_' .. interval_idx .. '.png', im_to_save)
    image.save('./results/MNIST/image_grids/last.png', im_to_save)
    to_save.confusion[interval_idx] = confusion
    to_save.GAN_count[interval_idx] = GAN_count
    torch.save('./results/MNIST/stream/confusions.t7', to_save)
    if interval_idx%10==0 then
      torch.save('./models/progress/MNIST_generators/interval_' .. interval_idx .. '_DCGAN.t7', GAN)
      torch.save('./models/progress/MNIST_stream_classifier.t7', C_model)
    end
  end
end
