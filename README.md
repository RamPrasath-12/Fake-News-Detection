# Fake News Detection

This project uses machine learning and natural language processing to detect fake news articles.  
It includes data preprocessing, feature extraction, model training, and prediction scripts.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/RamPrasath-12/Fake-News-Detection.git
cd Fake-News-Detection
```

---

### 2. Set Up Python Virtual Environment

```bash
python -m venv .myenv
```

**Activate the environment:**

- On Windows:
  ```bash
  .myenv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source .myenv/bin/activate
  ```

---

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

### 4. Add Data Files

Download or obtain the following files:
- `Fake.csv`
- `True.csv`

Place them in the `data/raw/` directory:
```
data/raw/Fake.csv
data/raw/True.csv
```

You can download the dataset from [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

---

### 5. Run the Project Scripts

**a. Merge and preprocess data:**
```bash
python src/merge_raw.py
python src/preprocess.py
```

**b. Feature extraction:**
```bash
python src/features.py
```

**c. Train the model:**
```bash
python src/train.py
```

**d. Make predictions:**
```bash
python src/predict.py
```

---

### 6. (Optional) Use Jupyter Notebooks

```bash
pip install notebook
jupyter notebook
```

---

### 7. Deactivate the Environment (when done)

```bash
deactivate
```




