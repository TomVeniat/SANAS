# SANAS: Stochastic Adaptive Neural Architecture Search

Implementation of **SANAS** (see [paper on arXiv ](https://arxiv.org/abs/1811.06753)), a model able to dynamically adapt deep architectures at test time for efficient sequence classification.

# Installation:
 - Create an environment with Python 3.6
 - `pip install -r requirements.txt`
   
# Speech commands dataset:
 - Download the Speech command v0.01 [archive](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz).
 - Extract the dataset and give the extracted folder path as `root_path` argument (defaults to `./data/speech_commands_v0.01`).
 - Implementation of the Speech Commands data processing is based on [honk](https://github.com/castorini/honk), credits goes to the authors!
 - Speech Commands dataset paper on [arXiv](https://arxiv.org/abs/1804.03209).
# Exemple run :

### Without mongo connection:
 - `python main.py with adam speech_commands gru kwscnn static=True use_mongo=False ex_path=<path_to_save_location>/runs` 
 - If no `ex_path` is specified, logs and models will be saved under `./runs`

### With mongo connection:
 - Create json file containing the required connection informations:

 ```json
 {
  "user": "Me",
  "passwd": "MySecurePassword",
  "host": "localhost",
  "port": "27017",
  "db": "sanas",
  "collection": "runs"
}
```
 - `python main.py with adam speech_commands gru kwscnn static=True use_mongo=False mongo_config_path=<path_to_config>/mongo_config.json`
 - `mongo_config_path` defaults to `./resources/mongo_credentials.json`
 
### Without Visdom :
 - `python main.py with adam speech_commands gru kwscnn static=True use_visdom=False` 

### With Visdom :
 - Visdom will connect to `localhost:8097` by default. To specify the server, create a config file:
 ```json
 {
  "server": "http://localhost",
  "port": 8097
}
```
- `python main.py with adam speech_commands gru kwscnn static=True visdom_config_path=<path_to_config>/vis_config.json` 


# Data :

### Implementing a new dataset:

The `__get_item__(self, idx)` of a dataset should return a tuple `(x,y)` with:
- `x` of size `seq_len x feature_dims`. For example, `feature_dims` for traditional images is `(C,H,W)`
- `y` of size `seq_len`.

It is possible to use the [PadCollate](https://github.com/TomVeniat/AdaptiveSequenceClassification/blob/master/src/commons/pytorch/data/collate.py#L20) class in the dataloader to pad each sequence to the length of the longest one in the sampled batch.
