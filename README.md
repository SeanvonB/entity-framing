# Selective Coference Resolution

This is the code associated with *Coreference-Based Entity Framing for Narrative Extraction* by Sean von Bayern and Luna Peck. All work was completed in Jupyter Notebooks using Google Colab, and configurations have been saved as reported in the paper. Trained models are too large for GitHub but are available upon request. If you are simply interested in the **selective coference resolution** function itself, this can be found in the standalone `utils.py` file.

## Requirements

It is strongly encouraged that you create separate virtual environements for training and coreference resolution due to unresovable package conflicts between these scripts.

For **training**:
```
conda create -n train
conda activate train
pip install -r requirements_train.txt
```

For **coreference**:
```
conda create -n coref
conda activate coref
pip install -r requirements_coref.txt
```
