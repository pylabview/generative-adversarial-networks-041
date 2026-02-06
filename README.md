1. This study ran the simulation on a 16" Mac Pro with the following specifications:

**ProductName:            macOS**
**ProductVersion:         26.2**
**BuildVersion:           25C56**
Hardware:

    Hardware Overview:
    
      Model Name: MacBook Pro
      Model Identifier: Mac16,5
      Model Number: Z1FW00086LL/A
      Chip: Apple M4 Max
      Total Number of Cores: 16 (12 performance and 4 efficiency)
      Memory: 128 GB
      System Firmware Version: 13822.61.10
      OS Loader Version: 13822.61.10
      Serial Number (system): LK6R49T0L7
      Hardware UUID: 98D180CC-6F0B-5D1A-9C88-16144F795A04
      Provisioning UDID: 00006041-000C099E3460801C
      Activation Lock Status: Enabled

2. This setup installs TensorFlow successfully with a few tweaks specific to Apple Silicon machines:

   Workflow:
   
   ```bash
   #### requirements.txt for MacOS Silicone M4 ####
   jupyterlab
   pandas
   seaborn
   requests
   scikit-learn
   polars
   duckdb
   pyarrow
   shap
   ipywidgets
   torch
   dill
   torchvision
   tensorflow==2.18.*
   tensorflow-metal==1.2.0
   ############################
   ```
   
   ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r req/requirements.txt
   #### Checking tensorflow installation
   python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
   
   2.18.1
   [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
   ```
   
   
   
   