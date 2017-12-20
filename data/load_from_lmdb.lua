local DataLoader = paths.dofile('./data.lua')
local data = DataLoader.new(opt.nThreads, 'lsun', opt)
