# MiCADO - Scaling Optimizer with Machine Learning Support

## Requirements 
Python >= 3.6 . 
flask . 
ruamel.yaml . 
numpy . 
scikit-learn . 
pandas . 


## Start program 
From project root run  
```python optimizer.py --cfg path/to_config_file```

## REST API 
Use `curl` for sending requests, for example for POST requests:  
```curl -X POST http://127.0.0.1:5000/optimizer/init --data-binary @path/to_init_file```

