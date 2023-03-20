# MutComputeX: A self-supervised 3D-Residual Neural Network for protein-X interface engineering

## X = nucleic acids, glycans, ligands, cofactors, other proteins



## To generate predictions on Norbelladine 4O-methyltransferase: 
### Norbelladine dataset and the MutComputeX models are made publicly available in an [S3 bucket](https://mutcomputex.s3.amazonaws.com) and must first be downloaded. 
#### To download the model weights and datasets (requires aws cli):
- `$ cd models && ./download_models.sh`
- `$ cd data/norbelladine_4OMTase/boxes/ && ./download_dataset.sh`

### Set the PYTHONPATH to the root directory of the repository:
- `$ export PYTHONPATH=$(pwd)`

### Run inference on Norbelladine 4O-methyltransferase computational structure: 
- `$ cd scripts && python generate_norbelladine_predictions.py`

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
- `$ pip install -r requirements.txt`

