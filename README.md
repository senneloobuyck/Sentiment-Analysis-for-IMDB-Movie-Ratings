# Sentiment Analysis Case – Text Classification Demo

## Project Overview

This project implements a **sentiment analysis pipeline** for classifying textual reviews using a **pre-trained NLP model**. The solution supports **single-text predictions** through a web interface as well as **batch prediction on CSV files**, returning both **sentiment labels** and **confidence scores**.

The case demonstrates model training and deployment via an interactive interface.


---

## Models 

Two BERT-models, fine-tuned on sentimental datasets, are benchmarked: 
- DistilBERT, fine-tuned on SST-2 dataset (general sentiment)
- RoBERTa large, fine-tuned on 15 different datasets (reviews, tweets, etc.)


---

## Project Structure

```
.
├── data/
│   ├── IMDB-movie-reviews-GT.csv
├── app.py
├── training.ipynb
├── requirements.txt
└── README.md
```



---

## Running the Project

### Install dependencies

```
pip install -r requirements.txt
```

### Launch the demo application

```
python app.py
```

The application will start locally and open a browser-based interface.
