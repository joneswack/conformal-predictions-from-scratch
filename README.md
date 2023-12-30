# Conformal Predictions from Scratch

The aim of this repository is to implement some popular Conformal Prediction methods from scratch using pure [NumPy](https://numpy.org/). The correctness is then verified with reference implementations from libraries like [MAPIE](https://mapie.readthedocs.io/en/stable/). All of this is done with illustrative examples and thorough explanation of the code.

Why do we reimplement things from scratch? Because we want to understand the methods from ground up in order to use them with confidence in future projects where stakes may be high. This understanding is also needed for researchers who may think about coming up with novel methods on their own. Therefore, the purpose of this repository is educational.

## 🕵️‍♀️ What are Conformal Predictions?

Conformal Predictions have become a very popular way of equipping machine learning models with predictive uncertainty estimates. One reason for this is that they are model-agnostic meaning that previously trained (or even deployed) models can be enhanced with predictive uncertainties post hoc. Another advantage is that Conformal Predictions come with theoretical guarantees on how much of the future data to be predicted will actually fall inside the predictive bands (generally assuming exchangeability of the data). This may seem to good to be true, but makes a lot of sense when understanding how the predictions are "conformalized".

A great list of references and tutorials on the topic can be found [here](https://github.com/valeman/awesome-conformal-prediction).

## 📝 The Structure of this Repository

The repository contains one Jupyter notebook per prediction method. One prediction method means one way to obtain predictive error bars. All notebooks are located in the *notebooks* folder and new notebooks will be added over time.

Notebook Overview:

### 📈 Regression

1) [Split Conformal Prediction](https://github.com/joneswack/conformal-predictions-from-scratch/blob/main/notebooks/regression/1-Split-Conformal-Prediction.ipynb): Conformal Predictions with fixed prediction bands obtained via (the default) absolute residual scores.
2) [Locally Weighted Conformal Prediction](https://github.com/joneswack/conformal-predictions-from-scratch/blob/main/notebooks/regression/2-Locally-Weighted-Conformal-Prediction.ipynb): Conformal Predictions with variable prediction bands obtained via residual normalised scores.