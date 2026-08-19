# Date Fruit Classification Using ANN

## 📌 Project Overview

This project uses an **Artificial Neural Network (ANN)** built with **PyTorch** to classify different varieties of date fruits based on their physical, shape, and color-related features.

The model performs **7-class classification** using the Date Fruit dataset and achieves approximately **94.44% test accuracy**.

## 🎯 Objective

The main objective is to develop a machine learning model that can automatically identify the variety of a date fruit from its extracted numerical features.

The seven classes are:

* BERHI
* DEGLET
* DOKOL
* IRAQI
* ROTANA
* SAFAVI
* SOGAY

## 📊 Dataset

The dataset contains **34 input features** describing different characteristics of date fruits, including:

* Area
* Perimeter
* Major Axis
* Minor Axis
* Eccentricity
* Solidity
* Convex Area
* Extent
* Aspect Ratio
* Roundness
* Compactness
* Shape Factors
* RGB color statistics
* Standard deviation
* Skewness
* Kurtosis
* Entropy
* Wavelet-based features

The target variable is:

```text
Class
```

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* PyTorch
* Jupyter Notebook

## 🔄 Machine Learning Pipeline

```text
Date Fruit Dataset
        ↓
Data Loading
        ↓
Data Validation
        ↓
Feature / Target Separation
        ↓
Label Encoding
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
PyTorch Tensor Conversion
        ↓
DataLoader
        ↓
Artificial Neural Network
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Classification Accuracy
```

## 🧹 Data Preprocessing

### 1. Handling Missing Values

The dataset was checked for null values using:

```python
df.isnull().sum()
```

No missing values were found.

### 2. Label Encoding

The categorical class labels were converted into numerical labels using `LabelEncoder`.

### 3. Feature Scaling

`StandardScaler` was used to normalize the input features.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler was fitted only on the training data to avoid data leakage.

## 🧠 ANN Architecture

The neural network consists of:

```text
Input Layer
34 Features
     ↓
Fully Connected Layer
34 → 64
     ↓
ReLU
     ↓
Fully Connected Layer
64 → 64
     ↓
ReLU
     ↓
Output Layer
64 → 7
```

### Model Configuration

| Component      | Configuration    |
| -------------- | ---------------- |
| Input Features | 34               |
| Hidden Layer 1 | 64 neurons       |
| Hidden Layer 2 | 64 neurons       |
| Activation     | ReLU             |
| Output Classes | 7                |
| Loss Function  | CrossEntropyLoss |
| Optimizer      | Adam             |
| Batch Size     | 32               |
| Epochs         | 100              |

## ⚙️ Training

The model was trained using the Adam optimizer and Cross-Entropy Loss.

```python
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
```

The training process uses:

1. Forward propagation
2. Loss calculation
3. Backpropagation
4. Gradient calculation
5. Parameter update

## 📈 Results

The trained ANN achieved:

### **Test Accuracy: 94.44%**

The training loss decreased significantly during training, indicating that the neural network successfully learned patterns from the dataset.

## 🔍 Model Evaluation

The model predicts the class with the highest output logit:

```python
_, predicted = torch.max(outputs, 1)
```

Accuracy is calculated as:

```text
Accuracy = Correct Predictions / Total Predictions × 100
```

Future evaluation can include:

* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Report

## 📁 Project Structure

```text
Date-Fruit-Classification-ANN/
│
├── DateFruit_Dataset.csv
├── Date_Fruit_ANN.ipynb
├── README.md
└── requirements.txt
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/SHAIKAMANBASHA/Date-Fruit-Classification-ANN.git
```

### 2. Navigate to the project

```bash
cd Date-Fruit-Classification-ANN
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Date_Fruit_ANN.ipynb
```

and execute the cells sequentially.

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
numpy
pandas
scikit-learn
torch
matplotlib
seaborn
jupyter
```

## 🔮 Future Improvements

* Add confusion matrix visualization
* Add precision, recall, and F1-score
* Perform hyperparameter tuning
* Compare ANN with Random Forest, SVM, and other classifiers
* Add a prediction interface for new date fruit samples
* Save and deploy the trained PyTorch model

## 👨‍💻 Author

**Aman Basha**

B.Tech – Computer Science and Engineering

## ⭐ Project Highlights

* 7-class date fruit classification
* 34 numerical input features
* PyTorch-based ANN
* Feature standardization
* 100 training epochs
* **94.44% test accuracy**
