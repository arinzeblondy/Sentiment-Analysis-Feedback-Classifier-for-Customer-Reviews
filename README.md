# Sentiment-Analysis-Feedback-Classifier-for-Customer-Reviews

This project is focused on building a machine learning-powered solution that helps companies analyze and understand customer sentiments from reviews and social media feedback in real-time.

>  Built with TF-IDF, Logistic Regression & Streamlit  
>  Dataset: Airline Twitter Sentiment (6,000+ real tweets)


 Project Title

Feature Engineering + Model Training For Sentiment Analysis & Feedback Classifier for Customer Reviews

Problem Statement

How can companies identify, classify, and act on customer sentiment (positive, neutral, negative) automatically — to improve service quality, customer satisfaction, and brand reputation?

Solution Overview

We built a machine learning pipeline that:

- Cleans and preprocesses real-world customer tweets
- Transforms text using TF-IDF vectorization
- Trains a Logistic Regression model to classify sentiment
- Deploys the model as an interactive web app using Streamlit

Dataset

Source:Twitter Airline Sentiment Dataset  
- 6,000+ labeled tweets
- Sentiments: Positive, Neutral, Negative
- Features: tweet text, airline, user feedback reason, etc.

Technologies Used

 Tool                     Purpose
Python                    Core programming
Pandas, Scikit-learn      Data processing & modeling
TF-IDF                    Feature extraction
Logistic Regression       Classification model
Streamlit                 Web app deployment 
jupyter Notebook          Exploratory analysis 

Model Performance

-TF-IDF + Logistic Regression
- Accuracy: ~78%
- Best performance on clear positive/negative cases
- Slight confusion between neutral vs. negative (expected in opinion-based data)

Streamlit App Features

- Clean and modern UI
- Paste any review or tweet
- Click Analyze to classify
- Displays prediction with:
  - Emoji feedback
  - Color-coded result
  - Sentiment label (POSITIVE, NEUTRAL, NEGATIVE)

 Use Cases

- Customer Service Analysis  
- Social Media Monitoring  
- Brand Reputation Tracking  
- Airline, Fintech, and E-commerce Review Monitoring



 How to Run Locally

 Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-app.git
   cd sentiment-analysis-app/app
