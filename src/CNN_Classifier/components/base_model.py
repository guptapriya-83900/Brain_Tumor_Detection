import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from CNN_Classifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    
    def get_base_model(self):
        # Adjust the input shape for grayscale
        input_tensor = tf.keras.layers.Input(shape=(self.config.params_image_size[0], self.config.params_image_size[1], 1))  # (300, 300, 1) for grayscale input
        
        # Convert grayscale to 3-channel by replicating the grayscale channel
        x = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))(input_tensor)
        
        # Load EfficientNetB3 with no top and ImageNet weights, but adjust input_tensor to have 3 channels
        base_model = tf.keras.applications.EfficientNetB3(
            include_top=self.config.params_include_top,
            weights='imagenet',  # Load with ImageNet weights
            input_tensor=x  # Pass the modified input tensor
        )
        
        # Since EfficientNetB3 expects three channels, we replicate the single grayscale channel three times
        self.model = tf.keras.models.Model(inputs=input_tensor, outputs=base_model.output)

        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_layers, learning_rate,optimizer_name,loss_function, metrics):
         # Fine-tuning: freeze layers except the first few
        for layer in model.layers[:freeze_layers]:
            layer.trainable = True  # Unfreeze the first few layers

        for layer in model.layers[freeze_layers:]:
            layer.trainable = False  # Freeze the remaining layers

        # Add global pooling and dense layer for classification
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax" if classes > 2 else "sigmoid"  # Softmax for multi-class, sigmoid for binary
        )(x)

         # Dynamically get the optimizer class from tf.keras.optimizers
        optimizer_class = getattr(tf.keras.optimizers, optimizer_name)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=optimizer_class(learning_rate=learning_rate),
            loss=loss_function, 
            metrics=metrics
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_layers=self.config.params_freeze_layers,  # Unfreeze the first few layers
            learning_rate=self.config.params_learning_rate,
            optimizer_name=self.config.params_optimizer,  # Use optimizer name from config
            loss_function=self.config.params_loss,  # Use loss function from config
            metrics=self.config.params_metrics  # Use metrics from config
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    
