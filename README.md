# Implement DATSAN

## Dependency:

- dgl
- pandas
- numpy
- sklearn

## Run

- donwload Elliptic dataset from [kaggle](https://kaggle.com/ellipticco/elliptic-data-set)
- unzip the dataset into a raw directory, such as /home/Elliptic/elliptic_bitcoin_dataset/
- make a new dir to save processed data, such as /home/Elliptic/processed/
- run train.py by:

```python
python train.py --raw-dir /home/Elliptic/elliptic_bitcoin_dataset/ --processed-dir /home/Elliptic/processed/
```

