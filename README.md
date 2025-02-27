# ESM2-650M CryptoBench model
This repository contains a model trained using the smaller ESM2-650M protein language model (3B version of the model was used in the Cryptobench study). Also, this model uses PyTorch framework instead of the Tensorflow framework. Check out the [original repository for the whole study](https://github.com/skrhakv/CryptoBench).

## Run the prediction
An example how to run the prediction can be found in `run-prediction.py`:
```
python3 run-prediction.py
```

## Install
Before running the model, install torch version `2.2.1`:
```
python3 -m pip install -r requirements.txt
```

## License
This repository is licensed under the [MIT license](https://github.com/skrhakv/smaller-cryptobench-model/blob/master/LICENSE).
