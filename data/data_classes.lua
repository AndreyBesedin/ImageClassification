local data_classes = {
  LSUN = {
    'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room',
    'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower',
  },
  MNIST = {
    'zero', 'one', 'two', 'three', 'four', 
    'five', 'six', 'seven', 'eight', 'nine',
  },
  CIFAR10 = {
    'cat', 'dog', 'airplane', 'automobile', 'truck',
    'horse', 'frog', 'bird', 'ship', 'deer'
  },
  Other = {
    'bedroom', 'bridge', 'church_outdoor', 'classroom', 'kitchen','tower',
  },
}

--function get_data_classes(dataset)
  return data_classes[opt.dataset]
--end
