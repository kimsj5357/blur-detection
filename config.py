from pprint import pprint

class Config:
    num_classes = 5
    img_size = 512
    num_workers = 8
    test_num_workers = 8

    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # preset
    net = 'vgg16'

    # training
    epoch = 14


    use_adam = False
    use_chainer = False
    use_drop = False

    load_path = None


    def _parse(self):
        state_dict = self._state_dict()

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
