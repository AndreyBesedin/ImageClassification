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
  lr = 0.001,
  data_folder = '/home/besedin/workspace/Data/LSUN/data_lmdb',
  nThreads = 6,
  imSize = 224,
  batchSize = 24,
  loadSize = 256,
  fineSize = 224,
  gpu = 1,
  epoch_nb = 100,
  testing = 'real',
  train_batch_real = false,
  totalClasses = 10, -- Total nb of classes in stream, basically unknown but since we use static datasets as stream, let's say we know it... 
  data_size = 2048
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
        
--classes = torch.FloatTensor(opt.pretrainedClasses);  -- Initializing classes to the start of the stream
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
path_to_testset = './subsets/LSUN/validation/testset_5k_per_class_1.t7'
--path_to_testset = './subsets/LSUN/100k_images_10_classes/pca_real_test.t7'
testset = torch.load(path_to_testset)
print('\nTESTSET LOADED, SIZE: ' .. testset.data:size(1)); 

---------------------------------------------------------------------------------------------------------
-- INITIALIZING MODELS
---------------------------------------------------------------------------------------------------------

--C_model = torch.load('./models/classifiers/LSUN_real_data_classifier.t7')
print('\nINITIALIZING CLASSIFICATION MODEL'); C_model = init_classifier_LSUN(opt.data_size, 10, opt)
print('\nMODELS INITIALIZED, START TRAINING'); sleep(2)

p, gp = C_model:getParameters()
---------------------------------------------------------------------------------------------------------
-- TRAINING
---------------------------------------------------------------------------------------------------------
function create_dataset(data_classes, ratios, scenario)
  local dataset = {}
  dataset.data = torch.FloatTensor(1000000, opt.data_size)
  dataset.labels = torch.FloatTensor(1000000)
  for idx_class = 1, #ratios do
    print('Loading class ' .. data_classes[idx_class])
    dataset.labels[{{1 + (idx_class-1)*100000, idx_class*100000}}]:fill(idx_class)
    local orig_data = torch.load('./subsets/LSUN/100k_images_10_classes/' .. data_classes[idx_class] .. '_reconstructed_bn.t7')
    --local orig_data = torch.load('./subsets/LSUN/100k_images_10_classes/pca_' .. data_classes[idx_class] .. '_real.t7')
    --local gen_data = torch.load('./subsets/LSUN/100k_images_10_classes/' .. data_classes[idx_class] .. '_gen.t7')
    dataset.data[{{1 + (idx_class-1)*100000, idx_class*100000},{}}] = orig_data:float()
    --if scenario == 'orig' then
    --  return dataset
    --end
    if ratios[idx_class]<1 then
      if scenario == 'mixed_gen' then
        print('Filling with generated data')
        dataset.data[{{1 + (idx_class-1)*100000, 100000*(1-ratios[idx_class]) + (idx_class-1)*100000},{}}] = gen_data[{{1, 100000*(1-ratios[idx_class])},{}}]
      elseif scenario == 'mixed_noise' then
        print("Filling with noise")  
        dataset.data[{{1 + (idx_class-1)*100000, 100000*(1-ratios[idx_class]) + (idx_class-1)*100000},{}}]:normal(0.35,200)
      end
    end
  end
  return dataset
end

function make_noise_dataset(data_classes, class_size)
  local dataset = {}
  dataset.data = torch.FloatTensor(5*class_size, 2048)
  dataset.labels = torch.FloatTensor(5*class_size)
  for idx_class = 1, 5 do
    print('Loading class ' .. data_classes[idx_class])
    dataset.labels[{{1 + (idx_class-1)*class_size, idx_class*class_size}}]:fill(idx_class)
    dataset.data[{{1 + (idx_class-1)*class_size, idx_class*class_size},{}}]:normal(idx_class, 1/idx_class)
  end
  return dataset
end

dataset = create_dataset(data_classes, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 'orig')
--dataset = make_noise_dataset(data_classes, 10000)
--testset = make_noise_dataset(data_classes, 1000)
confusion_test = {}
opt.batches_per_epoch = 1e+1
confusion_test[0] = test_classifier(C_model, testset); print(confusion_test[0])
--confusion_test[0] = test_classifier(C_model, dataset); print(confusion_test[0])

maxTotalValid = 0
for epoch = 1, 10 do
  C_model, _ = train_classifier(C_model, dataset, opt)
  print('Finished mini-epoch nb ' .. epoch)
  confusion_test = test_classifier(C_model, testset); print(confusion_test)
  if confusion_test.totalValid > maxTotalValid then maxTotalValid = confusion_test.totalValid; best_model = C_model:clone();end --torch.save('./best_model.t7', best_model) end
--    torch.save('results/batch_training/LSUN_real.t7', confusion_test)
end
