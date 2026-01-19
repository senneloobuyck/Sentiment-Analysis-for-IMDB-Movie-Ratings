# Sentiment Analysis Case – Text Classification Demo

## Project Overview

This project implements a **sentiment analysis pipeline** for classifying textual reviews using a **pre-trained NLP model**. The solution supports **single-text predictions** through a web interface as well as **batch prediction on CSV files**, returning both **sentiment labels** and **confidence scores**.

The case demonstrates model training and deployment via an interactive interface.


---

## Models 

Two BERT-models, fine-tuned on sentimental datasets, are benchmarked: 
- DistilBERT, fine-tuned on SST-2 dataset (general sentiment) and NYU Glue (general NLP benchmark)
- RoBERTa large, fine-tuned on 15 different datasets (reviews, tweets, etc.)


---

## Project Structure

```
.
├── data/
│   ├── IMDB-movie-reviews-GT.csv
│   ├── IMDB-movie-reviews-only.csv
├── app.py
├── training.ipynb
├── requirements.txt
└── README.md
```



---

## Running the Project

### Install dependencies (Python 3.12.9)

```
pip install -r requirements.txt
```

### Launch the demo application

```
python app.py
```

The application will start locally and open a browser-based interface. You can use the `IMDB-movie-reviews-only.csv` file to test the batch prediction on CSV file and compare with the ground-truth `IMDB-movie-reviews-GT.csv` file.
