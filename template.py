import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "CNN_Classifier"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"


]


for filepath in list_of_files:
    filepath = Path(filepath)             
    filedir, filename = os.path.split(filepath)


    if filedir !="":                                #Check if filedir is not empty
        os.makedirs(filedir, exist_ok=True)         #Make a directory and here we are stating "exist_ok=True" because if this directory is already present than the directory won't be created again
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0): #Checkinf if file exist or not
        with open(filepath, "w") as f:                                     #Creating file
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")


'''
Path(): This method will return full file path.
os.path.split(): This will return two parameters one is file directory and another is filename
Example:
Code:
from pathlib import Path
import os
path="US/Priya/main.py"
print(Path(path))
print(os.path.split(path))

Output:
US/Priya/main.py
('US/Priya', 'main.py')

os.makedirs(): This methos is used for creating directory. And we set "exist_ok=True" because if that directory already exists
we don't need to create it again

'''