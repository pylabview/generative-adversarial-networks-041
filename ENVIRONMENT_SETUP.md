# Reproducing the CIFAR-10 GAN Experiment (TensorFlow)

This document explains how to reproduce the TensorFlow/Keras GAN experiment (CIFAR-10) in the same environment used for the original run.

---

## 1) Tested Environment

### Hardware
- **Model:** 16-inch MacBook Pro (Model Identifier: `Mac16,5`)
- #### **Chip:** Apple M4 Max
  
  - **Total cores:** 16 (12 performance + 4 efficiency)
- **Memory:** 128 GB

### Operating System
- **macOS:** 26.2  
- **Build:** 25C56

> **Note (recommended):** Consider removing/redacting identifiers like serial number, hardware UUID, and provisioning UDID if this repo is public.

---

## 2) Python Environment Setup (Apple Silicon / Metal)

This setup uses a dedicated virtual environment and installs TensorFlow with Metal GPU support for Apple Silicon.

### Create & activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3) Dependencies

```bash
# Core / notebooks
jupyterlab
pandas
requests
scikit-learn
polars
duckdb
pyarrow
shap
ipywidgets
dill

# Visualization (optional)
seaborn

# Deep Learning (this experiment)
tensorflow==2.18.*
tensorflow-metal==1.2.0

# NOTE:
# torch/torchvision are NOT required for this TensorFlow GAN experiment.
# Keep them only if you use them elsewhere in the course repo.
# torch
# torchvision

```



## 4) Install Requirments

```bash
pip install -r req/requirements.txt
```

## 5) Verify TensorFlow + GPU (Metal)

Run the folowing to confirm TensoFlow is installed and detects GPU

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

### Expected or similar output
2.18.1
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 6) Run experiment

In the root application directory, open a Jupyter Lab session and run the [experiment nb](Assigment_4_Generative_Adversarial_Networks.ipynb) 

```bash
jupyter lab
```

