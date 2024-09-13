import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from CNN_Classifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path  # Path to the saved model
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Excludes the channel dimension (e.g., [224, 224])
            batch_size=self.config.params_batch_size,        # Number of images per batch
            interpolation="bilinear"                        # Method for resizing images
        )


        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )


        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Directory where the image data is stored
            subset="validation", 
            color_mode="grayscale",                 # Use the validation split (20% of data)
            shuffle=False,                        # No need to shuffle validation data
            **dataflow_kwargs                     # Apply target size and batch size settings
        )


        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,               # Rotate images randomly up to 40 degrees
                horizontal_flip=True,            # Flip images horizontally
                width_shift_range=0.2,           # Shift images horizontally by up to 20%
                height_shift_range=0.2,          # Shift images vertically by up to 20%
                shear_range=0.2,                 # Apply shear transformations to the images
                zoom_range=0.2,                  # Zoom in or out randomly up to 20%
                **datagenerator_kwargs           # Apply the common data preprocessing settings (e.g., rescaling)
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Directory where the image data is stored
            subset="training",
            color_mode="grayscale",                   # Use the training split (80% of data)
            shuffle=True,                        # Shuffle training data to ensure randomness
            **dataflow_kwargs                    # Apply target size and batch size settings
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        self.model.fit(
            self.train_generator,               # Training data generator
            epochs=self.config.params_epochs,    # Number of epochs (how many times the model sees the entire dataset)
            steps_per_epoch=self.steps_per_epoch,  # Number of training steps per epoch
            validation_steps=self.validation_steps, # Number of validation steps per epoch
            validation_data=self.valid_generator,   # Validation data generator
            callbacks=callback_list                 # List of callbacks (e.g., saving the model, logging metrics)
        )

        # Save the trained model after the training process is complete
        self.save_model(
            path=self.config.trained_model_path,  # Save path for the final model
            model=self.model                      # The trained model
        )
