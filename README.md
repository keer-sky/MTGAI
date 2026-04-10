# MTGAI
Transformer-based Multi-task Multitasking generative AI

This project implements a Transformer-based multi-task deep learning model for time series regression (future value prediction) and classification (sequence category recognition). The model adopts a multi-task learning framework with a task interaction module for feature sharing, supports dynamic task weight adjustment.
Project Structure
.
├── dataset.py           # Dataset class: data loading, normalization, augmentation  
├── model.py             # Model definitions: positional encoding, task interaction, Transformer  
├── trainer.py           # Trainer: training loop, validation, early stopping, scheduler  
├── data_utils.py        # Data splitting and saving/loading utilities  
├── utils.py             # Helper functions: sample data generation, checkpoint loading  
├── main.py              # Main entry point: integrates all modules and runs training  
└── README.md            # This file  

Features
Multi-task learning: Simultaneously outputs regression predictions (future values) and classification results (sequence category).
Transformer encoder: Captures long-range dependencies in time series.
Task interaction module: Enhances information sharing between regression and classification via cross-attention and gating.
Weighted sampling: Uses WeightedRandomSampler to balance training data and mitigate class imbalance.
Cross-entropy loss: Standard cross-entropy loss for classification, with class weight support.
Uncertainty weighting (optional): Automatically learns task weights for regression and classification.
Data augmentation: Adds noise and scaling during training to improve generalization.
Early stopping & model saving: Saves the best model based on validation loss.

Requirements

    Python 3.8+

    PyTorch 1.9+

    pandas, numpy, scikit-learn, scipy

Install dependencies

pip install torch pandas numpy scikit-learn scipy openpyxl

Usage

1.Adjust configuration parameters
Modify the parameters in the fixed_main() function inside main.py:

sequence_length = 8        # Input sequence length (number of past time steps)
prediction_length = 52     # Forecast horizon (number of future steps to predict)
batch_size = 32
epochs = 200
data_dir = 'data'          # Directory to save split datasets (train.csv, val.csv)
model_dir = 'models'       # Directory to save model checkpoints

2. Run training

python main.py

Training Strategy

Loss functions: SmoothL1Loss for regression, CrossEntropyLoss for classification (class weights supported).

Task weighting: Two modes:

uncertainty_weighting=True: Automatically learns task weights based on homoscedastic uncertainty.

False: Manual piecewise weights (first 30 epochs favor classification, later balanced).

Optimizer: AdamW with cosine annealing learning rate scheduler (initial lr=0.0001).

Early stopping: Stops if validation loss does not improve for 25 consecutive epochs.

License

This project is for research and educational purposes only.

