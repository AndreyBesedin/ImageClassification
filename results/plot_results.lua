require 'gnuplot'
epochs = {1, 2, 3, 5, 8, 15, 23, 31, 40, 50}

for idx = 1, #epochs do
  filename = 'step1_accuracies_generation' .. epochs[idx] .. '.t7'
  data = torch.load(filename)
  gnuplot.figure()
  gnuplot.plot({{'trainset', data[1]:squeeze()},{'testset1', data[2]:squeeze()},{'testset2', data[3]:squeeze()}})
  gnuplot.title('generation ' .. epochs[idx])
  gnuplot.grid(true)
  gnuplot.plotflush()
end
