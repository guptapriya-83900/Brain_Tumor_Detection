import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from CNN_Classifier.entity.config_entity import PrepareCallbacksConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config


    
    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        # Use Pathlib's `/` operator instead of os.path.join
        tb_running_log_dir = self.config.tensorboard_root_log_dir / f"tb_logs_at_{timestamp}"

        # Convert Pathlib Path to string for TensorBoard callback
        return tf.keras.callbacks.TensorBoard(log_dir=str(tb_running_log_dir))
    

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True
        )


    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]
