from CNN_Classifier.constants import *
from CNN_Classifier.utils.common import read_yaml, create_directories
from CNN_Classifier.entity.config_entity import DataIngestionConfig  #Added this line

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,         
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    #This method extracts the data ingestion-related configuration from the YAML file.
    def get_data_ingestion_config(self) -> DataIngestionConfig:   #This is for getting the all the configuration related to our data 
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,                                     #For storing all the values in the variable
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config