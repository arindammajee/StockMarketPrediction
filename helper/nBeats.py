from kerasbeats import NBeatsModel

class NBeats_Model:
    def __init__(self):
        self.model = None
        self.model_summary = None
        self.history = None
        self.batch_size = 16
        self.sc = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.train_mse = None
        self.val_mse = None
        self.test_mse = None

        
    def buildModel(self, input_shape = (10, 1), lstmNeurons=[8, 16, 8], dropOuts=[], hyperParam=None):
        nbeats = NBeatsModel()
        nbeats.build_layer()
        nbeats.build_model()
        self.model = nbeats.model
        
