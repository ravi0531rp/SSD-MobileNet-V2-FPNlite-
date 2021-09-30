class Config:
    def __init__(self):
        self.PATH_TO_CKPT = './saved_model/'  # give the path to exported saved_model.pb
        self.PATH_TO_LABELS = '.label_map.pbtxt' # path to label_map.ptxt
        self.TEST_MODE = False
