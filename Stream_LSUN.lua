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
dofile('./sup_functions.lua')
-------------------------------------------------------------------------------------------------------
-- ADVANCED OPTIONS FOR TRAINING
-------------------------------------------------------------------------------------------------------
opt = {
  lr = 0.0005,
  display = true,
  data_folder = '/home/besedin/workspace/Data/LSUN/data_lmdb',
  nThreads = 6,
  initClassNb = 4, -- Number of already pretrained classes in the model
  pretrainedClasses = {3, 4, 5},
--  pretrainedClasses = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
  maxClassNb = 5,      -- Maximum nb of classes in any stream interval
  usePretrainedModels = true,
  imSize = 224,
  batchSize = 16,
  loadSize = 256,
  fineSize = 224,
  interval_size = {200, 300},
  bufferSize = 50, -- Number of batches in the buffer
  gpu = 1,
  dropout = 0,
  epoch_nb = 1,
  testing = 'real',
  continue_training = false,
  start_interval = 100,
  init_pretrained = false,
  train_batch_fake = false,
  train_batch_real = false,
  totalClasses = 10, -- Total nb of classes in stream, basically unknown but since we use static datasets as stream, let's say we know it... 
}
if opt.display then disp = require 'display' end
opt.noise_vis = torch.CudaTensor(16, 100, 1, 1)
opt.noise_vis = opt.noise_vis:normal(0,1)

-- Full list of available classes
data_classes = {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 
                      'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'}    
opt.history_per_class = torch.zeros(#data_classes)
for idx = 1, #opt.pretrainedClasses do
    opt.history_per_class[opt.pretrainedClasses[idx]] = 5e+6
end
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

---------------------------------------------------------------------------------------------------------
-- INITIALIZING TRAINING AND PARAMETERS
---------------------------------------------------------------------------------------------------------
        
if not DATA then DATA = initialize_loaders(opt) end
interval_is_over = true;
GAN_count = torch.zeros(10); local buffer_count = torch.zeros(10)

classes = torch.FloatTensor(opt.pretrainedClasses);  -- Initializing classes to the start of the stream
-- Parameters for the training step zero:

buffer, buffer_count = init_buffer(opt)

classif_criterion = nn.ClassNLLCriterion()
GAN_criterion = nn.BCECriterion()
GAN_criterion = GAN_criterion:cuda()

optimState_GAN = {}
for idx = 1, 10 do
  optimState_GAN[idx]= {}
  optimState_GAN[idx].D = {learningRate=0.000005, beta1 = 0.5}
  optimState_GAN[idx].G = {learningRate=0.000005, beta1 = 0.5}
end
optimState = {
  learningRate = opt.lr,
  weightDecay = 1e-4,
}

config = {
  learningRate = opt.lr,
  weightDecay = 1e-4,
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

-- Step zero: Testing on just pretrained GANs for initial classes:
if opt.train_batch_fake then
  opt_zero = deepcopy(opt); opt_zero.bufferSize = opt.bufferSize*10;
  --buffer_zero, buffer_count_zero = init_buffer(opt_zero)
  --buffer_zero = complete_buffer(buffer_zero, buffer_count_zero, GAN, opt_zero)
  confusion_test = {}
  confusion_test[0] = test_classifier(C_model, testset); print(confusion_test[0])
  for epoch = 1, opt.epoch_nb do
    buffer_zero, buffer_count_zero = init_buffer(opt_zero)
    buffer_zero = complete_buffer(buffer_zero, buffer_count_zero, GAN, feature_extractor, opt_zero)
    confusion_test[epoch] = {}
    for idx = 1, 1 do
      C_model, confusion = train_classifier(C_model, buffer_zero, opt_zero)
      confusion_test[epoch] = {confusion}
    end
    torch.save('results/batch_training/LSUN_fake.t7', confusion_test)
  end
end

if opt.train_batch_real then
  confusion_test = {}
  confusion_test[0] = test_classifier(C_model, testset); print(confusion_test[0])
  for epoch = 1, opt.epoch_nb do
    for idx = 1, 5 do
      C_model, confusion = train_classifier(C_model, trainset, opt)
      confusion_test[idx+(epoch-1)*5] = confusion
    end
    torch.save('results/batch_training/LSUN_real.t7', confusion_test)
  end
end

-- opt_zero = deepcopy(opt); opt_zero.bufferSize = opt.bufferSize*5;
-- buffer_zero, buffer_count_zero = init_buffer(opt_zero)
-- buffer_zero = complete_buffer(buffer_zero, buffer_count_zero, GAN, feature_extractor, opt_zero)
-- for epoch = 1, 1 do
--   C_model = train_classifier(C_model, buffer_zero, opt_zero)
-- end

if opt.continue_training then 
  to_save = torch.load('./results/LSUN/stream/confusions.t7') 
else
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
  to_save.interval_idx = 0
  to_save.optimState_GAN = optimState_GAN
  to_save.history_per_class = opt.history_per_class
end
local Stream = true
function clearGAN(GAN)
  for idx =1, #GAN do
    GAN[idx].G:clearState()
    GAN[idx].D:clearState()
  end
  return GAN
end


if not to_save.optimState_GAN then to_save.optimState_GAN = optimState_GAN end
if not to_save.history_per_class then to_save.history_per_class = opt.history_per_class end
for idx_class = 1, 10 do
  batch_orig = DATA[idx_class]:getBatch(opt.batchSize)
  batch_orig = rescale_3D_batch(batch_orig:float(), 64)
  batch_gen = generate_data(GAN[idx_class].G, opt.batchSize)
  disp.image(batch_orig, {win=idx_class, title=opt.full_data_classes[idx_class] .. '_orig'})
  disp.image(batch_gen, {win=10+idx_class, title=opt.full_data_classes[idx_class] .. '_gen'})
end
while Stream do
  collectgarbage()
  if interval_is_over == true then 
    print('\nInterval ' .. to_save.interval_idx .. ' is over, starting next')
    to_save.interval_idx = to_save.interval_idx + 1
    interval, classes = get_new_interval(classes, opt) -- fill in the interval with ordered classes of batches from stream
    to_save.intervals.duration[to_save.interval_idx] = interval:size(1)
    to_save.intervals.classes[to_save.interval_idx] = classes
    print('New classes: '); print(classes:reshape(1,classes:size(1)))
    batch_idx = 1
    interval_is_over = false
  end
  -- Vizualization
  
  local current_class = interval[batch_idx]
  batch_idx = batch_idx + 1
  batch_orig = DATA[current_class]:getBatch(opt.batchSize)
  --print('RECEIVED DATA FROM CLASS ' .. current_class)
  if to_save.history_per_class[current_class] < 5e+6 then -- change for more sophisticated criterion later
    GAN[current_class], errD, errG = train_GAN(GAN, current_class, rescale_3D_batch(batch_orig:float(), 64), to_save.optimState_GAN[current_class])
    print('Class ' .. current_class .. ', errD = ' .. errD .. ', errG = ' .. errG)
  end
  to_save.history_per_class[current_class] = to_save.history_per_class[current_class] + opt.batchSize
  local batch_features = feature_extractor:forward(batch_orig:cuda())
  
  -- Filling in the buffer
  buffer_count[current_class] = buffer_count[current_class] + 1
  xlua.progress(buffer_count:sum(), opt.bufferSize*classes:size(1))
  GAN_count[current_class] = GAN_count[current_class] + 1
  buffer[{{current_class},{1 + (buffer_count[current_class]-1)*opt.batchSize, buffer_count[current_class]*opt.batchSize},{}}] = batch_features:clone():float()
  if buffer_count[current_class] == opt.bufferSize then
    print('Collected enough data. Samples distribution by class: '); print(buffer_count:reshape(1,10)) 
    buffer = complete_buffer(buffer, buffer_count, GAN, feature_extractor, opt)
    buffer_temp = {}
    buffer_temp.data = buffer.data:clone()
    buffer_temp.labels = buffer.labels:clone()    
    for idx_class = 1, 10 do
      batch_orig = DATA[idx_class]:getBatch(opt.batchSize)
      batch_orig = rescale_3D_batch(batch_orig:float(), 64)
      batch_gen = generate_data(GAN[idx_class].G, opt.batchSize)
      disp.image(batch_orig, {win=idx_class, title=opt.full_data_classes[idx_class] .. '_orig'})
      disp.image(batch_gen, {win=10+idx_class, title=opt.full_data_classes[idx_class] .. '_gen'})
    end
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
    to_save.confusion[to_save.interval_idx] = confusion
    to_save.GAN_count[to_save.interval_idx] = GAN_count
    torch.save('./results/LSUN/stream/confusions.t7', to_save)
    GAN = clearGAN(GAN)
    C_model:clearState()
    if to_save.interval_idx%1==0 then
      torch.save('./models/progress/LSUN_generators/interval_' .. to_save.interval_idx .. '_DCGAN.t7', GAN)
      torch.save('./models/progress/LSUN_stream_classifier.t7', C_model)
    end
    break
  end
end
