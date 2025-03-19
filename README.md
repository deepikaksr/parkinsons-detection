# Parkinson's Disease Detection

This project is designed to detect early signs of Parkinson’s disease using machine learning. It analyzes speech patterns from biomedical voice measurements and predicts whether an individual has Parkinson’s disease or not.

## Dataset

- Source: UCI Machine Learning Repository
- Contains 195 voice recordings from 31 individuals (23 with Parkinson’s, 8 healthy)
- Features: 22 numerical voice measurements such as frequency, jitter, and shimmer
- Target Variable: 1-Parkinson’s detected, 0-Healthy individual

## Project Structure

- **data/** - Contains the dataset file
- **parkinsons_detection.py** - Main script for training and predicting
- **README.md** - Project documentation

## Installation

1. Clone the repository:
git clone https://github.com/deepikaksr/parkinsons-detection.git cd parkinsons-detection

## Usage
To run the model, execute: python3 parkinsons_detection.py