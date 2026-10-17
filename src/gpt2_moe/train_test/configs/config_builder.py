# Basic Imports
import os
import yaml
from pathlib import Path



class ExperimentConfig():

    # Define constant values
    TASKS = ["add_sub", "mul_div", "alphabetic", "spell", "parity", "first_last"]
    BASE_DIR = os.path.join(Path.cwd(), 'configs/cf_files')

    print(BASE_DIR)

    def __init__(self, name:str):

        # Define the base data storage
        self.data:dict = None
        # Load yaml file
        self.load_yaml(name)

        # Define stages as None
        self.exp_cofig:list[dict] = None
        self.iter_idx:int = 0

        # Build the stages
        self.build_stages()
        self.stop_id = len(self.exp_cofig['stages'])

        #print('Aspecto de la exp_cofig', self.exp_cofig)

    def get_model_params(self)->dict:
        """ 
            Function aimed to return model 
            configuration  and init mode

            Returns:
                model_config (dict):
                    The model configuration
        """
        return self.data["MODEL_ATTR"]
    
    def get_train(self)->dict:
        """  
            Function aimed to return train 
            configuration

            Returns:
                train_config(dict):
                    Dictionary containing train 
                    configuration
        """
        return self.data["TRAIN"]

    def __iter__(self):
        return self

    def __next__(self):
        # Check index
        if self.iter_idx >(self.stop_id - 1):
            # Reboot iter idx
            self.iter_idx = 0
            raise StopIteration
        # Return next item 
        value = self.exp_cofig['stages'][self.iter_idx]
        # Increase idx
        self.iter_idx+=1
        return value

    def build_stages(self)->None:
        """  
            Function aimed to build stages for 
            our stages_based training
            
            Remember that stages have the following
            format:

            STAGES:
                name: 'E0'
                init_mode: random
                stages: 
                     - name: "test_1"
                      groups:
                     - router
                     - ffn
                      expert: null
                      tasks: null
                      loss: ce

        """ 
        # Get stages data
        exp_cofig = self.data['EXPCONFIG']    

         # Set the init mode and load modality
        self.name = exp_cofig["name"]
        self.init_mode = exp_cofig["init_mode"]
        self.load_modality = exp_cofig["load_modality"]
        experts_mode = exp_cofig["train_experts_individually"]

        # Add expert stages
        expert_stages = [{"name": f"expert_{i}", "groups": ["experts"], "expert": i, "tasks": [task], "loss": "ce"}
                    for i, task in enumerate(self.TASKS)]
        # Modify base stages
        exp_cofig["stages"] = (expert_stages  if experts_mode else [])  + exp_cofig["stages"]

        # Set values
        self.exp_cofig = exp_cofig

       

    @staticmethod
    def file_checker(name:str)->str:
        """
            Function aimed to check file existance
            In case the selected file does not exist,
            a default file will be returned

            Params:
                name(str):
                    The yaml name to validae

            Returns:
                final_path(str):
                    The path for the final file
        """
        # Define mock path
        mock_path = os.path.join(ExperimentConfig.BASE_DIR, name)
        # Check if it exists
        if os.path.exists(mock_path):
            return mock_path

        print(f'The seeked file ({name}) does not exist')
        return  os.path.join(ExperimentConfig.BASE_DIR, 'base.yaml')

    def load_yaml(self, name:str)->dict:
        """  
            Function aimed to conduct yaml file
            loading

            Params:
                name(str):
                    The name of the file to load

            Return:
                data(dict):
                    The dictionary containing retrieved params

            
        """

        # Get path
        path = self.file_checker(name)

        # Load data
        with open(path, mode = 'r') as file:
            self.data = yaml.safe_load(file)
