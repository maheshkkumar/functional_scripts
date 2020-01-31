#### Functional Scripts

This repository contains the common utility scripts.


1. **intermediate_features.py**

This script extracts the intermediate layer-wise features from a network.  
Note: You can update the script to accommodate custom architectures.

##### Usage
```python
python intermediate_features.py --help

usage: intermediate_features.py [-h] -dp DATA_PATH -sd SAVE_DIR

optional arguments:
  -h, --help            show this help message and exit
  -dp DATA_PATH, --data_path DATA_PATH
                        Path of the image/folder
  -sd SAVE_DIR, --save_dir SAVE_DIR
                        Path to save the features

```