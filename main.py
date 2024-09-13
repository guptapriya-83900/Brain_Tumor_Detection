from src.CNN_Classifier.logger import logging
from CNN_Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from CNN_Classifier.pipeline.stage_02_base_model import PrepareBaseModelTrainingPipeline
from CNN_Classifier.pipeline.stage_03_training_model import ModelTrainingPipeline
from CNN_Classifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise e

STAGE_NAME = "Base Model stage"
try:
   logging.info(f"****************************************")
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   base_model = PrepareBaseModelTrainingPipeline()
   base_model.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise e

STAGE_NAME = "Training Model stage"
try:
   logging.info(f"****************************************")
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   train_model = ModelTrainingPipeline()
   train_model.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise e

STAGE_NAME = "Evaluation stage"
try:
   logging.info(f"****************************************")
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   evaluate_model = EvaluationPipeline()
   evaluate_model.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise e