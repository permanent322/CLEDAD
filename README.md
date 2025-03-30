The repository for the paper **CLEDAD: Contrastive Learning Enhanced Conditional Diffusion for Time Series Anomaly Detection**.



This is a refactored version of the resulting code of the paper for ease of use. Follow these steps to copy each cell in the result table.

### Results

![image-20250330182230788](./results/result.png)

### Installation

This code needs Python-3.8 or higher.

```
conda create --name cledad python=3.8 -y
conda activate cledad
pip3 install -r requirements.txt
```

### Datasets

Download public datasets used in our experiments:

```
python src/utils/download_data.py [dataset-name]
```

Options of [dataset-name]: msl, smd, smap, swat and psm.

###  Run Experiments

To run the experiments on different dataset, you can just run the following command:

```
python src/experiments/train_test_anomaly_detection.py --dataset [dataset-name] --device [device] --seed [seed] --anomaly_ratio [anomaly_ratio] --config [config-name]
```

The values of [dataset-name], [seed] and [anomaly_ratio] used in our experiments are available in our paper.

You can modify the config file to train and test with different parameters. The config file is located in `src/config` directory. For different datasets, we used different `[dataset-name].yaml` configurations

### Acknowledgement

The code for this library is referenced from the following repositories, in particular data download and processing:

TSDE:  https://github.com/ZinebSN/TSDE

