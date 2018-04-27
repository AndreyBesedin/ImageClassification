function init_G(opt)
  local G = nn.Sequential()
  local ngf = opt.GAN_params.ngf; local nc = opt.GAN_params.nc
  G:add(SpatialFullConvolution(100, ngf * 8, 4, 4))
  G:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
  -- state size: (ngf*8) x 4 x 4
  G:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
  G:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
  -- state size: (ngf*4) x 8 x 8
  G:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
  G:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
  -- state size: (ngf*2) x 16 x 16
  G:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
  G:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
  -- state size: (ngf) x 32 x 32
  G:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
  G:add(nn.Tanh())
  G:apply(weights_init)
  return G
end

function init_D()
  local D = nn.Sequential()
  D:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
  D:add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf) x 32 x 32
  D:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
  D:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf*2) x 16 x 16
  D:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
  D:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf*4) x 8 x 8
  D:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
  D:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf*8) x 4 x 4
  D:add(SpatialConvolution(ndf * 8, 1, 4, 4))
  D:add(nn.Sigmoid())
  -- state size: 1 x 1 x 1
  D:add(nn.View(1):setNumInputDims(3))
  D:apply(weights_init)
  return D
end

