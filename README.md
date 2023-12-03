# MutComputeX: A self-supervised 3D-Residual Neural Network for protein-X interface engineering

## X = nucleic acids, glycans, ligands, cofactors, other proteins



## To generate predictions on Norbelladine 4O-methyltransferase:
The models can be executed either natively, or from within a docker container.  Regardless, the Norbelladine data must be download from an s3 bucket.
### Download dataset
The following command, when run from the root directory of this repo, will download the dataset to the appropriate location.

`$ cd data/norbelladine_4OMTase/boxes/ && ./download_dataset.sh`

### Run the Model (native)
All of the following commands must be run from the root directory of this repository.

To run the model natively, you must furst download the model with the following command:

`$ cd models && ./download_models.sh`

Then you must set the PYTHONPATH to the root directory of the repository with the following command:

`$ export PYTHONPATH=$(pwd)`

Finally, you can run the inference:

`$ cd scripts && python generate_norbelladine_predictions.py`

### Run the Model (docker)
The model and its dependencies are all bundled in the docker image.  To run the model within a docker container, first build the docker image with the following command:

`$ docker build -t mutcomputex:latest .`

Then run the following script:

`$ cd scripts && python generate_norbelladine_predictions_docker.py`

## System Requirements

### Hardware Requirements
Models were trained using AMD GPUs (MI50s) with tensorflow-rocm >= 2.9.x using 'channel first'.   
Channel first tensorflow models can only run on GPUs. Thus, an AMD GPU is required to genereate inferences.

### Software requirements
This package has been tested on Ubuntu 18.04 and 20.04 and requires:  
- python >= 3.7.x
- ROCM >= 5.1.x
- tensorflow-rocm >= 2.9.x
- pandas >= 1.4.x
- AWS cli >= 2.9.x (download data and models)

#### install requirements:
`$ pip install -r requirements.txt`

