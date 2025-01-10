# LAWGORYTHM: Multi-Agent AI Data Analyst for Judicial Cases

## Overview
LAWGORYTHM is an AI-powered web platform designed to simplify and enhance the analysis of judicial cases. By integrating advanced modules such as the **Petition Pipeline**, **ACT Module**, and **LawBot**, LAWGORYTHM provides a seamless solution for processing legal documents, performing advanced analysis, and offering interactive legal assistance.

---

## Features
- **Data Collection**: Automates the ingestion of legal documents.
- **Data Preprocessing**: Cleans and structures text data for efficient analysis.
- **AI-Powered Analysis**: Uses ML models for classification, summarization, and prediction.
- **Interactive Chatbot**: Provides legal insights and answers user queries in real-time.
- **Visualization**: Displays intuitive insights through charts and graphs.

---

## Table of Contents
1. [Installation](#installation)
2. [Architecture](#architecture)
3. [Key Components](#key-components)
   - [Petition Pipeline](#petition-pipeline)
   - [ACT Module](#act-module)
   - [LawBot](#lawbot)
4. [Usage](#usage)
5. [Deployment](#deployment)
6. [Contributing](#contributing)
7. [License](#license)

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Joshuathomas18/Multi-Agent--AI--Data--Analyst--for--Judicial--Cases_codered25.git
   ```

2. **Navigate to the Directory**:
   ```bash
   cd Multi-Agent--AI--Data--Analyst--for--Judicial--Cases_codered25
   ```

3. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Unix/MacOS
   venv\Scripts\activate     # On Windows
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Architecture

```
                          +----------------+
                          |   User Input   |
                          +--------+-------+
                                   |
                                   v
                          +----------------+
                          |   Petition     |
                          |   Pipeline     |
                          +--------+-------+
                                   |
                   +---------------+----------------+
                   |                                |
         +---------+----------+         +-----------+---------+
         |       ACT Module    |         |       LawBot        |
         +---------+----------+         +-----------+---------+
                   |                                |
                   +---------------+----------------+
                                   |
                          +--------+-------+
                          |   Database      |
                          +----------------+
```

---

## Key Components

### **1. Petition Pipeline**
Handles data preprocessing, classification, and summarization of legal documents.

#### Example Code:
```python
# Load dataset
import pandas as pd
df = pd.read_csv('final_judge_database.csv')

# Preprocess text
df['cleaned_text'] = df['text_column'].str.lower().str.replace(r'[^a-z0-9 ]', '')

# Train a classifier
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

### **2. ACT Module**
Performs advanced analysis on the processed data, including feature engineering and visualization.

#### Example Code:
```python
# Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])

# Train Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Visualize Features
import matplotlib.pyplot as plt
plt.barh(feature_names, rf_model.feature_importances_)
plt.show()
```

---

### **3. LawBot**
An AI chatbot that uses the outputs of the pipeline and ACT module to provide legal insights interactively.

#### Example Code:
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['input']
    response = "Processed legal data: " + user_input  # Replace with actual processing
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
```

---

## Usage

1. **Run the Petition Pipeline**:
   ```bash
   python pipeline.py
   ```

2. **Run the ACT Module**:
   ```bash
   python act_module.py
   ```

3. **Launch the Chatbot**:
   ```bash
   python lawbot.py
   ```

4. **Access the Web Interface**:
   Open `http://localhost:5000` in your browser.

---

## Deployment

1. **Deploy Backend**:
   - Use Flask for local hosting.
   - Deploy on Heroku or AWS for production.

2. **Host Frontend**:
   - Use Netlify or Vercel for static hosting.

3. **Integrate API**:
   Connect frontend to backend using REST APIs.

---



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
Special thanks to the contributors and open-source libraries that made this project possible.

