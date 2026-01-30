# Understanding Endogenous Data Drift in Adaptive Models with Recourse-Seeking Users Codebase

This codebase implements experiments for studying algorithmic fairness and recourse in machine learning models, particularly focusing on fair top-k selection methods and continual learning scenarios.

## 📁 Project Structure

```
├── Config/                    # Configuration files for different model types
│   ├── config.py             # Base logistic regression configuration
│   ├── MLP_config.py         # Multi-layer perceptron configuration
│   ├── continual_config.py   # Continual learning (logistic regression)
│   └── continual_MLP_config.py # Continual learning (MLP)
├── Dataset/                   # Dataset handling
│   ├── makeDataset.py        # Dataset creation and preprocessing
│   └── UCI_Credit_Card.csv   # Credit card dataset
├── Experiment_Helper/         # Utility functions and visualization
│   ├── helper.py             # Main helper class with visualization tools
│   └── auxiliary.py          # Auxiliary functions (weights, data updates, file saving)
├── Experiments/              # Main experiment implementations
│   ├── fair_topk*.py         # Fair top-k selection experiments
│   ├── topk*.py             # Standard top-k selection experiments
    ├── .... many other exerimnts
│   └── *_output/            # Generated results and visualizations
└── Models/                   # Machine learning models and algorithms
    ├── logisticRegression.py # Logistic regression implementation
    ├── MLP.py               # Multi-layer perceptron
    ├── recourse.py          # Algorithmic recourse methods
    └── synapticIntelligence.py # Continual learning algorithms
```

## 🔬 Experiment Types

### Fair Top-K Selection (`fair_topk_*.py`)

Implements the **Exp3** class that performs:

1. **Recourse generation** on dataset D
2. **Diverse-k labeling** using density-based sampling
3. **Model retraining** with updated dataset
4. **Fairness metrics calculation**

### Standard Top-K Selection (`topk_*.py`)

Implements the **Exp2** class for baseline comparisons without fairness constraints.

### Model Variants

- **Base models**: Logistic Regression (`fair_topk.py`, `topk.py`)
- **MLP models**: Multi-layer perceptron versions (`*_MLP.py`)
- **Continual Learning**: With catastrophic forgetting mitigation (`*_CL.py`, `*_DCL.py`)

## 🛠️ Key Components

### Helper Class (`Experiment_Helper/helper.py`)

- **Visualization**: PCA scatter plots, probability histograms, animated evolution
- **Metrics**: Balanced accuracy, JS divergence calculation
- **Animation**: Generates GIF visualizations of model evolution over rounds

### Core Algorithm (`fair_topk`)

The main experimental loop in the `update` method:

1. **Data sampling**: Updates training set with new samples
2. **Recourse application**: Selects fraction of rejected samples for recourse
3. **Fair labeling**: Uses KDE-based diversity sampling to select positive cases
4. **Model retraining**: Updates model with modified dataset
5. **Metrics tracking**: Calculates failure-to-recourse rates, balanced accuracy, etc.

## ⚙️ Configuration Parameters

Key experimental parameters (modifiable in each experiment file):

```python
THRESHOLD = 0.7        # Decision threshold (0.5, 0.7, 0.9)
RECOURSENUM = 0.5      # Fraction of rejected samples for recourse (0.2, 0.5, 0.7)
COSTWEIGHT = 'uniform' # Recourse cost weighting (uniform, log, inverse_gamma)
POSITIVE_RATIO = ...   # Target positive selection ratio
```

## ⚙️ Experiment

1. **Configure helper for model type**:

   - Go to `Experiment_Helper/helper.py`
   - Update the configuration import based on your model type:
     - For non continaul learning Logistic Regression: `from Config.config import test, train, sample`
     - For non continaul learning MLP: `from Config.MLP_config import test, train, sample`
     - For Continual Learning: `from Config.continual_config import test, train, sample` or `from Config.continual_MLP_config import test, train, sample`

2. **Configure parameters** in the desired experiment file

3. **Run the experiment** (change the .py file as you need):

   ```bash
   python Experiments/fair_topk_MLP.py
   ```

4. **View results**:
   - Animated GIFs in `*_output/` folders show model evolution
   - CSV files contain detailed metrics across rounds

## 📊 Output

Each experiment generates:

- **Animated visualization** (`.gif`): Shows data distribution evolution over training rounds
- **Metrics CSV**: Tracks failure-to-recourse rates, costs, balanced accuracy, etc.
- **Timestamped files**: Format: `{RECOURSENUM}_{THRESHOLD}_{POSITIVE_RATIO}_{COSTWEIGHT}_{DATASET}_{timestamp}`

## 🎯 Research Focus

This codebase implements research on:

- **Algorithmic recourse** for rejected applicants
- **Fair top-k selection** with diversity constraints
- **Continual learning** scenarios with fairness preservation
- **Long-term fairness** dynamics in iterative decision systems

The experiments study how fairness interventions affect model performance and applicant outcomes over multiple rounds of decision-making and model updates.
