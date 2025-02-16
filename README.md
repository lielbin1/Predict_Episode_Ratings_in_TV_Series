# IMDb Episode Rating Prediction

This repository contains the code for predicting IMDb episode ratings using machine learning models and natural language processing (NLP) techniques. The study evaluates the impact of subtitle-based features alongside IMDb metadata.

## Overview
- **Dataset**: IMDb metadata and Hebrew subtitles. Available on Kaggle:  
  [Predicting IMDb Ratings with Hebrew Subtitles](https://www.kaggle.com/datasets/lielbinyamin1997/predicting-imdb-ratings-with-hebrew-subtitles)
- **NLP Techniques**: Named Entity Recognition (NER) using DictaNER and subtitle embeddings from DictaLM2.0-instruct.
- **Machine Learning Models**: Random Forest, XGBoost, LightGBM, and MLP.
- **Feature Engineering**: TF-IDF, sentiment analysis, and PCA for dimensionality reduction.

## Key Findings
- IMDb metadata remains the strongest predictor of episode ratings.
- Subtitle-based features contribute minimally to improving prediction accuracy.
- PCA improved model performance by reducing subtitle embedding dimensions.

## Acknowledgments
This research utilized IMDb datasets, DictaLM, DictaNER, and machine learning frameworks such as Scikit-learn, XGBoost, and LightGBM.

---

