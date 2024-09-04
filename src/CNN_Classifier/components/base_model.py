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
        # Step 1: Load the ResNet50 model with pre-trained weights, adjusting for grayscale input (1 channel)
        input_tensor = tf.keras.layers.Input(shape=self.config.params_image_size)  # (224, 224, 1) for grayscale input
        
        # Load the ResNet50 model without weights for the initial build (weights=None)
        self.model = tf.keras.applications.ResNet50(
            input_tensor=input_tensor,
            weights=None,  # Don't load pre-trained weights yet
            include_top=self.config.params_include_top
        )

        # Step 2: Load pre-trained ResNet50 model with ImageNet weights (3 channels)
        pre_trained_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)  # RGB input shape
        )

         # Step 3: Copy weights from the pre-trained model, except for the first layer
        for layer in self.model.layers:
            if layer.name != 'conv1_conv':  # Skip the first conv layer, which has different input shape
                try:
                    layer.set_weights(pre_trained_model.get_layer(layer.name).get_weights())
                except:
                    # Some layers may not have weights (like pooling or dropout), so we skip them
                    pass


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

    
