
# Toward Model Interpretability in Medical NLP: A Systematic Analysis of BioBERT

**LING380: Topics in Computational Linguistics Final Project**

**James Cross (j.cross@yale.edu) and Daniel Kim (d.j.kim@yale.edu), December 2021**


## How to run code 

Two options available to run the code; on Google colab and/or locally on your machine.

#### Option 1) Google Colab 

[Model training notebook](https://colab.research.google.com/drive/1uPIi-OVchs_8A-SNcQtLfwelr0ccsz19?usp=sharing)

[Model evaluation/analysis notebook](https://colab.research.google.com/drive/1Hfy58JvyPbx55lKKhQAzzrhJIbN_Io0j?usp=sharing)

#### Option 2) Local Machine 

Notebooks: You can run the `model_training.ipynb` or `model_evaluation.ipynb` notebooks on your local machine. Directory paths will have to be adjusted as needed.


## Code Organization

- `data`: contains medical transcriptions and other medical report data used in model fine-tuning and analysis, clinical stop words, and saved accuracy and entropy metrics during evaluation
- `notebooks`: 
> - `model_training.ipynb`: code to train and fine-tune BERT and BioBERT
> - `model_evaluation.ipynb`: code to run various model evaluations, visualize word importances, perform post-training clinical stopword masking, and other analyses
> - `scripts`: same functionality as in the notebooks, in executable python scripts / functions
- `models`: where checkpoints of the best performing BERT and BioBERT models after hyperparameter optimization are stored


## Dependencies

All packages needed to run the code are available in the default Google Colab environment (see documentation for full list), with the exception of huggingface (`transformers`), used for loading transformer models, and captum.ai (`captum`), which provides access for a variety of model interpretation tools. 

