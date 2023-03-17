# MutComputeX
## A self-supervsied 3D-Residual Neural Network capable of protein : non-protein interface engineering

## To generate predictions on Norbelladine 4O-methyltransferase: 
### Download the model weights and datasets:
- `$ cd models && ./download_models.sh`
- `$ cd data/norbelladine_4OMTase/boxes/ && ./download_dataset.sh`

### install requirements:
- `$ pip install -r requirements.txt`

### Set the PYTHONPATH to the root directory of the repository:
- `$ export PYTHONPATH=$(pwd)`

### Run inference: 
- `$ cd scripts && python generate_norbelladine_predictions.py`
