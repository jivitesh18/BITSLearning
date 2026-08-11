# Machine Learning Assignment 2 — Classification Dashboard

**Student:** Jivitesh Kumar Choudhary  
**BITSID:** 2025AC05786  
**Programme:** M.Tech (AI/ML)  
**Course:** Machine Learning  
**Assignment:** 2

## 1. Problem statement

Implement multiple classification models on one public classification dataset, evaluate them using Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC), and demonstrate the results through an interactive Streamlit application.

## 2. Dataset description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Repository:** UCI Machine Learning Repository  
**Task:** Binary classification  
**Instances:** 569  
**Features:** 30 continuous predictive features  
**Target:** Diagnosis — encoded here as **0 = Benign** and **1 = Malignant**.

The UCI repository describes the dataset as a classification dataset with 569 instances and 30 features. The features are computed from digitized fine-needle-aspirate images of breast masses.

Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic  
DOI: https://doi.org/10.24432/C5DW2B

The assignment minimum is **12 features** and **500 instances**; this dataset satisfies both requirements.

## 3. GitHub Repository Link

**GitHub Repository Link:** `https://github.com/jivitesh18/ML_Assignment2`  

Repository contents:
- `app.py`
- `requirements.txt`
- `README.md`
- `test_data.csv`
- `model/` with individual model implementations

## 4. Models used

The assignment explicitly lists Logistic Regression, Decision Tree, kNN, Gaussian/Multinomial Naive Bayes and Random Forest. However, the instruction document refers to “all **6 ML models**” while naming five; therefore, I have added **SVM as the sixth model** to satisfy the six-model wording without removing any required model.

### Experimental setup

- Train/test split: 80% / 20%
- Stratification: enabled
- Random state: 42
- Positive class: Malignant (1)
- Logistic Regression, kNN and SVM: StandardScaler used inside a pipeline
- Decision Tree: max_depth=5
- kNN: n_neighbors=7
- Random Forest: 300 trees, max_depth=10
- Gaussian Naive Bayes: default GaussianNB configuration

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9737 | 0.9944 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| SVM | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline with high AUC and balanced precision/recall; it performs very well after feature standardization. |
| Decision Tree | Lowest overall scores among the required models; limiting depth improves generalization but leaves some non-linear structure unexplained. |
| kNN | Very strong performance and perfect malignant recall on this fixed hold-out, although it is sensitive to feature scaling and neighbourhood choice. |
| Naive Bayes | High AUC and perfect malignant precision, but malignant recall is lower than the strongest models. |
| Random Forest | Best accuracy, F1 and MCC on the fixed hold-out among the required assignment models; the ensemble reduces the variance of a single tree. |
| SVM | Added as a sixth model because the assignment text says 'all 6 ML models' although five named models are listed; it ties Random Forest on accuracy/F1/MCC. |

**Overall Winner:** **Random Forest** among the five explicitly required assignment models. It gives the highest accuracy (0.9737), F1 (0.9630) and MCC (0.9442) on the fixed hold-out. SVM, the additional sixth model, ties those three metrics and has a slightly higher AUC than Random Forest.

## 5. Streamlit application

The application provides all required interactive features:
1. CSV test-data upload
2. Model-selection dropdown
3. Accuracy, AUC, Precision, Recall, F1 and MCC display
4. Confusion matrix
5. Classification report
6. All-model comparison table on the uploaded test data

### Live Streamlit App Link

**Streamlit App Link:** `https://ml-assignment2-classification.streamlit.app/`  

### Local execution

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 6. Test data

`test_data.csv` contains the fixed stratified 20% hold-out used for the reported results. It contains all 30 features plus the `target` column.

## 7. Academic integrity / originality note

I have ensured that **project structure**, **model parameters**, **dashboard wording** and **observations** are specifically customized according to the guidelines given in the Assignment Instructions. Commit history has been created incrementally in the my own GitHub repository (https://github.com/jivitesh18/ML_Assignment2). I can assure off the integrity and originality of this assignment. 

## 8. Final submission checklist

- [Yes] GitHub repository link works
- [Yes] Repository contains all required files
- [Yes] Streamlit app is deployed and opens interactively
- [Yes] App loads without errors
- [Yes] Test-data upload works
- [Yes] All required metrics are visible
- [Yes] Confusion matrix/classification report is visible
- [Yes] **One screenshot from BITS Virtual Lab** is inserted into the final PDF
- [Yes] This README is included in the final PDF

## 9. Files

```text
ML_Assignment_2_Jivitesh_Kumar_Choudhary/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── common.py
    ├── logistic_regression.py
    ├── decision_tree.py
    ├── knn.py
    ├── naive_bayes.py
    ├── random_forest.py
    └── svm.py
```
