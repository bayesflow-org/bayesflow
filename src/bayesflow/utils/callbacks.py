import keras


def detailed_loss_callback() -> keras.callbacks.Callback:
    """
    Provides detailed callback for loss trajectory for every training step.
    """
    loss_callback = keras.callbacks.LambdaCallback(
        on_train_batch_end=lambda batch, logs: logs["loss"], on_test_batch_end=lambda batch, logs: logs["val_loss"]
    )

    return loss_callback


def custom_loss_callback() -> keras.callbacks.Callback:
    """
    Provides customizable callback for loss trajectory
    """
    raise NotImplementedError
