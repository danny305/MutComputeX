# MutComputeX: A self-supervsied 3D-Residual Neural Network for protein-X interface engineering

## X = nucleic acids, glycans, ligands, cofactors, other proteins



## To generate predictions on Norbelladine 4O-methyltransferase: 
### Norbelladine dataset and the MutComputeX models are made publicly available in an [S3 bucket](https://mutcomputex.s3.amazonaws.com) and must first be downloaded. 
#### To download the model weights and datasets (requires aws cli):
- `$ cd models && ./download_models.sh`
- `$ cd data/norbelladine_4OMTase/boxes/ && ./download_dataset.sh`

#### 
### Install requirements:
- `$ pip install -r requirements.txt`

### Set the PYTHONPATH to the root directory of the repository:
- `$ export PYTHONPATH=$(pwd)`

### Run inference on Norbelladine 4O-methyltransferase computational structure: 
- `$ cd scripts && python generate_norbelladine_predictions.py`

## The models require AMD GPUs and tensorflow-rocm >= 2.9.x in order to generate inferences.
