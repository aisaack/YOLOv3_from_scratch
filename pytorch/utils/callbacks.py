
class Callback:
    def __init__(self, **kwargs):
        self.callbacks = []

    def on_train_bein(self, logs = None):
        pass

    def on_epoch_begin(self, hook, logs = None):
        pass

    def on_batch_begin(self, hook, logs = None):
        pass

    def on_predict_begin(self, logs = None):
        pass

    def on_predict_end(self, logs = None):
        pass

    def on_test_begin(self, logs = None):
        pass

    def on_test_end(self, logs = None):
        pass

    def on_batch_end(self, hook, logs = None):
        pass

    def on_epoch_end(self, hook, logs = None):
        pass

    def on_train_end(self, logs = None):
        pass
