# PHM-Data-Preprocessing

### Introduction

This codebase is aimed at preprocessing the PHM Rock Drill dataset and generate engineered-features which are to be later modelled using various classification algorithms in the [PHM-Classifiers](https://github.com/Data-Pundits/PHM-Classifiers) repository.

There are three python scripts which are part of a data-engineering pipeline and can be orchestrated by simply running the bash script - '*data-pipeline.sh*'.

### Software Installation

Please install the prerequisite Softwares and libraries in order to run the scripts in this repository successfully:

1. [Python 3](https://www.python.org/downloads/) or greater. (**Required**)
2. [Java (JVM) 8/11/17](https://www.java.com/en/download/) (Needed for PySpark library) (**Required**)
3. Python Libraries (Run the below *pip* commands in a **Terminal** or **Command-Prompt** window) (**Required**):
   * **pyspark**: `pip install pyspark==3.3.2`
   * **numpy**: `pip install numpy==1.21.5`
   * **jupyter-notebook**: `pip install jupyter`
   * **scipy**: `pip install scipy==1.10.1`
   * **pandas**: `pip install pandas==1.5.3`
   * **scikit-learn**: `pip install scikit-learn==1.2.1`
   * **matplotlib**: `pip install matplotlib==3.7.0`
   * **seaborn**: `pip install seaborn==0.11.2`
   * **xgboost**: `pip install xgboost==1.7.4`
4. [VS Code](https://code.visualstudio.com/download) or any other preferred Code editor (Optional).

### Datasets

* The raw datasets are included as part of this repository and can be found under the 'training_data/' path.
* The script generates intermediate stage ('*consolidated_stage/*' and '*consolidated_trimmed_stage/*') and final ('*feature_extracts/*') datasets. Some samples data files are also provided in this repository for reference.
* The final dataset (ʼ*feature_extracts/'*) will be required to run the various classifier notebooks available the [PHM-Classifiers](https://github.com/Data-Pundits/PHM-Classifiers) repository.

### How to Run the Scripts

First step is to make sure the '*training_data/'* folder exists as per the *datasource_config.py* file. Please modify this config file in case the training_data is located elsewhere. The scripts in this repository are designed to use the path provided in the config file.

The three scripts namely preprocessing.py, trimmer.py and feature-extraction.py are to be run in the same order and can be orchestrated by simply running the data-pipeline.sh bash script in a Terminal or Command-Prompt window.

The bash-script can be run by using the command 'data-pipeline.sh'.

If the individual python scripts are to be run, the following commands can be used:

* `python preprocessing.py`
* `python trimmer.py`
* `python feature-extraction.py`

> **Note**: The Python run command 'python' may not work in all cases and can be replaced and tried with 'py' or 'python3' depending on the individual system environment variables configuration for Python.
