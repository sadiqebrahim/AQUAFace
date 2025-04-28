class Config(object):
    def __init__(self, test_model_path='checkpoints/resnet18_110.pth',lfw_test_list='/data/Datasets/lfw/lfw_test_pair.txt'):
        self.env = 'default'
        self.backbone = 'resnet18'
        self.classify = 'softmax'
        self.num_classes = 14017
        self.metric = 'arc_margin'
        self.easy_margin = False
        self.use_se = True
        self.loss = 'con_loss'

        self.display = False
        self.finetune = False

        self.train_root = '/data/Datasets/webface/CASIA-maxpy-clean-crop-144/'
        self.train_list = '/data/Datasets/webface/train_data_13938.txt'
        self.val_list = '/data/Datasets/webface/val_data_13938.txt'

        self.test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
        self.test_list = 'test.txt'

        self.lfw_root = '/data/Datasets/lfw/lfw-align-128'
        self.lfw_test_list = lfw_test_list

        self.checkpoints_path = 'r100_cas'
        self.load_model_path = 'models/resnet18.pth'
        self.test_model_path = test_model_path
        self.save_interval = 10

        self.train_batch_size = 16 # batch size
        self.test_batch_size = 64

        self.input_shape = (1, 128, 128)

        self.optimizer = 'adam'

        self.use_gpu = True  # use GPU or not
        self.gpu_id = '0, 1'
        self.num_workers = 4  # how many workers for loading data
        self.print_freq = 100  # print info every N batch

        self.debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
        self.result_file = 'result.csv'

        self.max_epoch = 50
        self.lr = 1e-1  # initial learning rate
        self.lr_step = 10
        self.lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
        self.weight_decay = 5e-4
        self.momentum = 0.95
