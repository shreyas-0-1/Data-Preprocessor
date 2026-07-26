# Cross-Sectional Data Preprocessor

A reusable preprocessing utility for cross-sectional machine learning datasets built on top of pandas and scikit-learn.

The package supports:

- Train-test splitting
- Leakage-safe fitting
- One-hot encoding for nominal categorical variables
- Ordinal encoding with user-defined category order
- Standardization of numerical variables
- Unknown-category handling
- pandas DataFrame output with feature names
- scikit-learn-compatible `fit`, `transform`, and `fit_transform`
- Input validation and error handling

You can do this all in one without scikit-learn's confusing syntax and documentation.
---

## Installation

Clone the repository:

```bash
git clone https://github.com/shreyas-0-1/Data-Preprocessor.git
cd DataPreprocessor.py
pip install -r requirements.txt