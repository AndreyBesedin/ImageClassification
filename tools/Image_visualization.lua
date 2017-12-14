require 'cunn'
require 'cudnn'
require 'image'

function show_multiple_images(data, nb1, nb2)
--[[
  Image visualization function.
  Input: data - dictionary with field 'data' and 'labels'
            data.data - 4D torch Tensor of size [nSamples x nChannels x height x width]
            data.labels - 1D torch Tensor of size [nSamples]
       nb1 - number of raws to show
       nb2 - number of columns to show
  --]]
  if not nb2 then nb1 = 10; nb2=10; print('Number of images to show is not provided, taking default values') end
  assert(data, "Please make sure you provided data to function arguments")
  assert(data.data:size():size() == 4, "Please make sure you use correct image format (type: torch.Tensor; dim = [nb_samples, nb_channels, height, width])")
  nb_samples = data.data:size(1); nb_channels = data.data:size(2); h = data.data:size(3); w = data.data:size(4);
  local labels = torch.zeros(nb1, nb2)
  local res_image = torch.zeros(data.data:size(2), h*nb1, w*nb2):float()
  local ids = torch.randperm(data.data:size(1))[{{1, nb1*nb2}}]:long()
  for i1 = 1, nb1 do
    for i2 = 1, nb2 do
      res_image[{{},{1+(i1-1)*h, i1*h},{1+(i2-1)*w, i2*w}}] = data.data[{{ids[(i1-1)*nb2 + i2]}, {}, {},{}}]:reshape(nb_channels,h,w)
      labels[i1][i2] = data.labels[ids[(i1-1)*nb2 + i2]]
    end
  end
  print('\nVisualizing a ' .. nb1 .. 'x' .. nb2 .. ' table of data samples with labels: \n')
  print(labels) 
  if not image then require 'image' end
  image.display(res_image)
  return res_image
end

noise_init = torch.CudaTensor(100, 100, 1, 1):normal(0,1)

function generate_and_show(model_path, nb1, nb2)
  local noise
  if not noise_init then
    noise = torch.CudaTensor(nb1 * nb2, 100, 1, 1):normal(0,1)
  else
    noise = noise_init
  end
  local model = torch.load(model_path)
  local res = model:forward(noise); nc = res:size(2); s1 = res:size(3); s2 = res:size(4)
  local res_to_show = torch.CudaTensor(nc, s1*nb1, s2*nb2)
  for idx1 = 1, nb1 do
    for idx2 = 1, nb2 do
      res_to_show[{{}, {1 + (idx1-1)*s1, idx1*s1}, {1 + (idx2-1)*s2, idx2*s2}}] = res[{{idx1 + nb1*(idx2-1)}, {}, {}, {}}]:squeeze()
    end
  end
  image.display(res_to_show:float())
end
