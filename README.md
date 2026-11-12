<div align="center">

# 💼 Adult Census Income Prediction: Naive Bayes vs Random Forest

**Machine Learning coursework: predicting whether a person earns more than $50K a year from 1994 US Census data, with full cleaning, class balancing and model comparison.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-6A1B9A?style=flat-square)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-11557C?style=flat-square)

<img src="assets/marital_status_by_income.png" width="440" alt="Marital status distribution by income"/>
<img src="assets/age_by_income.png" width="440" alt="Age distribution by income"/>

</div>

---

## 📌 Overview

Uses the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult) (32,561 training and 16,281 test records) to classify income as **≤50K** or **>50K**, and compares a probabilistic model (**Gaussian Naive Bayes**) with an ensemble model (**Random Forest**).

## 🧹 Pipeline

1. **Load** the train / test splits directly from the UCI repository
2. **Exploratory analysis:** distributions of every feature split by income class, plus a correlation heatmap
3. **Cleaning**
   - `?` treated as missing and imputed with the column **mode**
   - Duplicates removed and text normalised to lowercase
   - `native-country` binarised (United States vs other)
   - `capital-gain − capital-loss` combined into a single **`total-capital`** feature
   - Redundant `fnlwgt` and `education` dropped (`education-num` kept)
4. **Label encoding** of categorical features
5. **SMOTE oversampling** to balance the training classes (24,594 records per class)
6. **Train and evaluate** both models on the official test set: classification reports, confusion matrices, ROC and precision-recall curves

## 📈 Results (test set, 16,227 records)

| Model | Accuracy | ROC-AUC | >50K Precision | >50K Recall | >50K F1 |
|---|:-:|:-:|:-:|:-:|:-:|
| Gaussian Naive Bayes | **83.4%** | 0.867 | **0.71** | 0.50 | 0.59 |
| Random Forest | 83.1% | **0.889** | 0.62 | **0.72** | **0.67** |

Overall accuracy is almost identical. **Random Forest ranks examples better (higher AUC) and finds far more of the high earners** (recall 0.72 vs 0.50). Naive Bayes is more conservative: fewer false alarms, but more missed high earners.

<p align="center">
  <img src="assets/cm_naive_bayes.png" width="400" alt="Confusion matrix: Naive Bayes on test data"/>
  <img src="assets/cm_random_forest.png" width="400" alt="Confusion matrix: Random Forest on test data"/>
</p>

<details>
<summary><b>Correlation heatmap</b></summary>
<p align="center"><img src="assets/correlation_heatmap.png" width="650" alt="Correlation heatmap of encoded features"/></p>
</details>

## 📂 Notebooks

| Notebook | Description |
|---|---|
| `notebooks/adult_income_classification.ipynb` | ⭐ **Final submission:** EDA, cleaning, SMOTE, Naive Bayes vs Random Forest, all evaluation plots |
| `archive/MLCW2.ipynb` | Second iteration with extra EDA (boxplots, pair plots, correlation heatmap, feature importance) |
| `archive/MLCW1.ipynb` | First iteration |

## 🚀 Running

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook notebooks/adult_income_classification.ipynb
```

The dataset is downloaded automatically from the UCI repository when the notebook runs.

---

<div align="center"><sub>Built by <a href="https://github.com/minudakaliyanage">Minudaka Liyanage</a></sub></div>
