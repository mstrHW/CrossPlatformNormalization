import tensorflow as tf
from keras.callbacks import Callback


class TestHistoryCallback(Callback):
    def __init__(self, log_dir, test_data, scoring_method):
        super(TestHistoryCallback, self).__init__()
        self.test_data = test_data
        self.log_dir = log_dir
        self.summary_writer = tf.summary.FileWriter(log_dir)
        self.scoring_method = scoring_method

    def on_train_begin(self, logs={}):
        self.scores = []

    def on_train_end(self, logs={}):
        self.summary_writer.flush()
        self.summary_writer.close()
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        score = self.scoring_method(self.test_data)
        summary = tf.Summary()
        summary.value.add(tag='test_loss_mae', simple_value=score[0])
        summary.value.add(tag='test_loss_r2', simple_value=score[1])
        self.summary_writer.add_summary(summary, epoch)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
