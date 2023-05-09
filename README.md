# SEQUER (SEQuence-Aware Explainable Recommendation)
 
This is our Pytorch implementation of the paper:

Ariza-Casabona, A., Boratto, L., Salamó, M. (2023). First Steps Towards Self-Explaining Sequence-Aware Recommendation, RECSYS'23.

Please cite our paper if you use this repository.

## Datasets

- Amazon Beauty
- Amazon Sports
- Amazon Toys
- Yelp

The preprocessed and filtered datasets are taken from the [P5 github repository](https://github.com/jeykigung/P5). For those interested in the original [Amazon](https://nijianmo.github.io/amazon/index.html) and [Yelp](https://www.yelp.com/dataset) datasets, refer to the lei's [Sentires-Guide](https://github.com/lileipisces/Sentires-Guide) on how to extract feature explanation quadruples.

In order to be able to use P5's datasets, they must be processed by process_data.py which converts them to our dataset format and performs a user temporal split: ````python3 process_data.py```` 

## Usage

0. Update utils/constants.py file with the BASE_PATH to store the data, results and checkpoint folders
1. Update scripts/sequer.sh with the --model-suffix corresponding to the configuration file you want to use. By default, this script runs the code on each dataset with 5 different seeds on a CUDA device.
2. Give necessary permissions to the script file: chmod +x scripts/sequer.sh
3. Run in background: ````nohup ./scripts/sequer.sh > sequer_log.txt &````

In case you want to run a single experiment:
````python3 main.py --model-name ${model} --model-suffix=${suffix} --dataset ${dataset} --fold 0 --seed ${seed} --cuda --log-to-file````

Additional arguments include: ````--test```` (to select a small subset of the training set for quick code debugging), ````--no-generate```` (to avoid generating the file with the predicted explanations), ````--load-checkpoint```` to evaluate the last generated checkpoint for that model configuration.

## Dependencies

The necessary Python packages required to run our code are:

````
- torch==1.8.1
- pandas==1.1.5
- numpy==1.18.0
- pickle5
````
