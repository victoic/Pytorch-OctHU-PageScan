class BaseDataset():
  def __init__(self, num_classes, name, dataset_train, dataset_test, trainArgs = None, testArgs = None, path = None, criterion=None, transform = None):
    self.num_classes = num_classes
    self.name = name
    assert dataset_train is not None and dataset_test is not None

    ## DEFINES DATASET PATHS
    aboslute_path = pathlib.Path(__file__).parent.parent.absolute() if path == None else path
    self.path = os.path.join(aboslute_path, "data", name)
    self.path_train = os.path.join(aboslute_path, "data", name) if trainArgs['train_path'] is None else os.path.join(aboslute_path, "data", name, trainArgs['train_path'])
    self.path_test = self.path_train if testArgs['test_path'] is not None else os.path.join(aboslute_path, "data", name, testArgs['test_path'])
    if not os.path.exists(self.path):
        os.makedirs(self.path)
    ## REMOVES PATH FROM ARGS
    testArgs.pop('test_path')
    trainArgs.pop('train_path')
    
    self.set_datasets(dataset_train, dataset_test, transform, trainArgs, testArgs)
    self.criterion = torch.nn.CrossEntropyLoss() if criterion == None else criterion
    

  def set_datasets(dataset_train, dataset_test, transform, trainArgs, testArgs):
    self.data_train = dataset_train(self.path_train, transform = transform, **trainArgs)
    self.data_test = dataset_test(self.path_test, transform = transform, train = False, **testArgs)

class PageScanDataset(Dataset):
    """Page Scan dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filenames = []
        self.root_dir = root_dir
        self.transform = transform
        for root, dirs, files in os.walk(root_dir):
          for file in files:
            if "_in" in file:
              self.filenames.append(os.path.join(root, file))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fn = self.filenames[idx]
        gt_fn = fn.replace("_in", "_gt")
        
        img = Image.open(fn)
        gt = Image.open(gt_fn)
        
        if self.transform:
          sample = [self.transform(img), self.transform(gt)]
        else:
          sample = [img, gt]

        return sample

class PageScan(BaseDataset):
  def __init__(self, **kwargs):
    self.milestones = [15, 30, 60]

    if "_DEF_HEIGHT" not in kwargs:
      _DEF_HEIGHT = 512
    if "_DEF_WIDTH" not in kwargs:
      _DEF_WIDTH = 512


    if "transform" in kwargs:
        transform = transform
    else:
        transform = transforms.Compose([
            torchvision.transforms.Resize((_DEF_HEIGHT,_DEF_WIDTH)),

            transforms.ToTensor()
        ])

    testArgs = {'test_path': 'validation'}
    trainArgs = {'train_path': 'train'}
    super().__init__(10, "PageScan", PageScanDataset, PageScanDataset, 
        testArgs = testArgs, trainArgs = trainArgs, transform=transform, criterion = self.dice_coef, **kwargs)
    print(f"PageScan dataset downloaded to {self.path}")

  def dice_coef(self, y_pred, y_true, smooth=1000.0):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return -(2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

  def set_datasets(self, dataset_train, dataset_test, transform, trainArgs, testArgs):
    self.data_train = dataset_train(self.path_train, transform = transform)
    self.data_test = dataset_test(self.path_test, transform = transform)