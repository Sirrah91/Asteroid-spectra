from tensorflow.keras.callbacks import EarlyStopping


class ReturnBestEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super(ReturnBestEarlyStopping, self).__init__(*args, **kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                print(f"\nEpoch {self.stopped_epoch + 1}: early stopping")
        elif self.restore_best_weights:
            if self.verbose > 0:
                print(f"Restoring model weights from the end of the best epoch: {self.best_epoch + 1}.")
            self.model.set_weights(self.best_weights)
