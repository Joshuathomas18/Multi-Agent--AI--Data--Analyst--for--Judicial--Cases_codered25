# <img src="image_4.jpg" alt="LAWGORITHM Logo" width="65" height="65">LAWGORITHM: Multi-Agent AI Data Analyst for Judicial Cases




## Overview
LAWGORITHM is an AI-powered web platform designed to simplify and enhance the analysis of judicial cases. By integrating advanced modules such as the **Petition Pipeline**, **ACT Module**, and **LawBot**, LAWGORITHM provides a seamless solution for processing legal documents, performing advanced analysis, and offering interactive legal assistance.
1. **Multi-Agent Collaboration**:
   - **Petition Pipeline**: Acts as the core processing unit, ingesting raw legal documents, preprocessing them, and applying machine learning models to classify and summarize case data. It creates structured outputs that are ready for further analysis or direct use in legal petitions.
   - **ACT Module**: Analyzes the processed data from the pipeline, performing advanced tasks such as feature engineering, legal clause tagging, and generating in-depth case summaries. This module is optimized to identify trends and correlations that might be overlooked by manual review.
   - **LawBot**: A conversational agent designed to interact directly with users. It uses the outputs of the Petition Pipeline and ACT Module to answer queries, provide legal insights, and generate draft petitions dynamically.

2. **Automated Petition Drafting**:
   - By combining the outputs from all modules, LAWGORYTHM can generate draft petitions tailored to specific cases. These drafts include relevant legal clauses, case references, and actionable insights, reducing the time lawyers spend on document preparation.

3. **Insight Extraction**:
   - The platform not only prepares petitions but also extracts actionable insights, such as identifying pivotal legal precedents or summarizing case histories. This helps lawyers make data-driven decisions more efficiently.

4. **Ease of Use**:
   - Lawyers can interact with the system through an intuitive web interface, allowing them to upload case files, query specific legal issues, or request drafts of petitions in real-time. The conversational capabilities of LawBot ensure seamless interaction without requiring technical expertise.

### Benefits for Legal Professionals
- **Efficiency**: Automates labor-intensive tasks, enabling lawyers to focus on strategy and argumentation.
- **Accuracy**: Reduces errors by leveraging AI-powered models for data processing and summarization.
- **Scalability**: Handles large volumes of case data, making it suitable for individual practitioners and large law firms alike.
- **Cost-Effectiveness**: Saves time and resources by automating repetitive workflows.

---

## Features
- **Data Collection**: Automates the ingestion of legal documents.
- **Data Preprocessing**: Cleans and structures text data for efficient analysis.
- **AI-Powered Analysis**: Uses ML models for classification, summarization, and prediction.
- **Interactive Chatbot**: Provides legal insights and answers user queries in real-time.
- **Visualization**: Displays intuitive insights through charts and graphs.



<p align="center">
  <img src="image_2.jpg" alt="LAWGORYTHM Logo" width="155" height="175">
</p>

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
   - [Petition Module](#petition-module)
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


![Alt Text](system-architecture.svg)                 



![Alt Text](image.png)
## Key Components

### **1. Petition Module**
The Petition Module serves as the foundational step in the pipeline. It focuses on ingesting, preprocessing, and classifying judicial case data to extract key legal insights.

#### **Step-by-Step Workflow**

1. **Data Loading**:
   - Import the judicial case dataset.
   - Use pandas to load the data into a DataFrame.
   ```python
   import pandas as pd
   df = pd.read_csv('final_judge_database.csv')
   ```

2. **Text Preprocessing**:
   - Clean the text data to make it suitable for model training.
   - Steps include converting text to lowercase and removing special characters.
   ```python
   import re
   def preprocess_text(text):
       text = text.lower()
       text = re.sub(r'[^a-z0-9 ]', '', text)
       return text

   df['cleaned_text'] = df['text_column'].apply(preprocess_text)
   ```

3. **Feature Extraction**:
   - Transform the text data into numerical features using vectorization techniques.
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   vectorizer = TfidfVectorizer()
   X = vectorizer.fit_transform(df['cleaned_text'])
   ```

4. **Train-Test Split**:
   - Split the dataset into training and testing sets for model evaluation.
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, df['target'], test_size=0.2, random_state=42)
   ```

5. **Model Training**:
   - Train a classification model, such as Logistic Regression, on the prepared dataset.
   ```python
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression()
   model.fit(X_train, y_train)
   ```

6. **Model Evaluation**:
   - Evaluate the trained model using metrics like accuracy and classification reports.
   ```python
   from sklearn.metrics import accuracy_score, classification_report
   y_pred = model.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, y_pred))
   print("Classification Report:
", classification_report(y_test, y_pred))
   ```

7. **Save Processed Data**:
   - Save the processed data and trained model for downstream use in the pipeline.
   ```python
   import joblib
   joblib.dump(model, 'petition_model.pkl')
   ```

This module outputs predictions and summarized insights, which feed into the subsequent ACT Module for deeper analysis.
Handles data preprocessing, classification, summarization, and key legal insights extraction.

#### Example Code:
```python
# Load dataset
import pandas as pd
df = pd.read_csv('final_judge_database.csv')

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text

df['cleaned_text'] = df['text_column'].apply(preprocess_text)

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
An AI chatbot that uses the outputs of the Petition and ACT modules to provide legal insights interactively.

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
![Alt Text](image_3.jpg)
## Usage

1. **Run the Petition Module**:
   ```bash
   python petition.py
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

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
Special thanks to the contributors and open-source libraries that made this project possible.
