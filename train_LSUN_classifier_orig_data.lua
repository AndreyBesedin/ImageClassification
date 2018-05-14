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
  data_folder = '/home/besedin/workspace/Data/LSUN/data_lmdb',
  nThreads = 6,
  imSize = 224,
  batchSize = 16,
  loadSize = 256,
  fineSize = 224,
  gpu = 1,
  dropout = 0,
  epoch_nb = 100,
  testing = 'real',
  train_batch_real = false,
  totalClasses = 10, -- Total nb of classes in stream, basically unknown but since we use static datasets as stream, let's say we know it... 
}

-- Full list of available classes
data_classes = {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 
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

---------------------------------------------------------------------------------------------------------
-- INITIALIZING TRAINING AND PARAMETERS
---------------------------------------------------------------------------------------------------------
        
if not DATA then DATA = initialize_loaders(opt) end
classes = torch.FloatTensor(opt.pretrainedClasses);  -- Initializing classes to the start of the stream
-- Parameters for the training step zero:

classif_criterion = nn.ClassNLLCriterion()

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
print('\nLOADING FEATURE EXTRACTOR'); feature_extractor = init_feature_extractor('./models/feature_extractors/resnet-200.t7')
print('\nMODELS INITIALIZED, START TRAINING'); sleep(2)

p, gp = C_model:getParameters()
---------------------------------------------------------------------------------------------------------
-- TRAINING
---------------------------------------------------------------------------------------------------------

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
