# Sentiment Analysis Case – Text Classification Demo

## 📌 Project Overview

This project implements a **sentiment analysis pipeline** for classifying textual reviews using a **pre-trained NLP model**. The solution supports single-text predictions through a web interface as well as **batch prediction on CSV files**, returning both **sentiment labels** and **confidence scores**.

The case demonstrates the full workflow from data preparation and model training to deployment via an interactive interface.

---

## 🎯 Objectives

- Train or fine-tune a sentiment analysis model on labeled review data  
- Provide a user-friendly interface for sentiment prediction  
- Support batch inference on CSV files  
- Return predictions together with confidence scores  
- Ensure modular, reusable, and production-oriented code structure  

---

## 🧠 Model & Approach

- **Task**: Binary sentiment classification (e.g. positive / negative)  
- **Frameworks**: Flair NLP  
- **Model**: Pre-trained sentiment classifier (fine-tuned on custom data)  
- **Evaluation**: Stratified train/validation split  

The model outputs:
- **Predicted sentiment label**
- **Confidence score** (model certainty)

---

## 🗂️ Project Structure

```
.
├── data/
│   ├── IMDB-movie-reviews.csv
├── train/
│   ├── IMDB-movie-reviews.csv
├── models/
│   └── sentiment-model/
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

---

## 📊 Data Format

### Input CSV (for batch prediction)

```
review
"This product is amazing"
"The service was terrible"
```

### Output CSV

```
review,prediction,confidence
"This product is amazing",POSITIVE,0.97
"The service was terrible",NEGATIVE,0.95
```

---

## 🏋️ Model Training

Training is performed using a **stratified dataset split** to ensure balanced sentiment classes.

---

## 🖥️ Application Interface

The project includes an **interactive web interface** built with **Gradio** that allows:

- Manual text input for single predictions  
- Uploading a CSV file for batch sentiment analysis  
- Display of predicted sentiment and confidence score  

This makes the solution suitable for both technical users and business stakeholders.

---

## 🚀 Running the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model (optional)

```
python train.py
```

### 3. Launch the demo application

```
python app.py
```

The application will start locally and open a browser-based interface.

---

## 👤 Author

Developed as part of a **Sentiment Analysis Case** for Sopra Steria.
