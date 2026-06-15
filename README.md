# End-to-End MLOps Pipeline using DVC, MLflow, FastAPI, GitHub Actions & AWS EC2

## Project Overview

This project demonstrates a complete Machine Learning Operations (MLOps) workflow, covering the journey of a machine learning model from training to cloud deployment.

The goal of the project is not just to train a machine learning model, but to show how a model can be:

* Version controlled
* Experiment tracked
* Saved and reused
* Exposed through an API
* Automated using CI/CD
* Deployed on AWS

This simulates how machine learning models are managed and deployed in real-world organizations.

---

## Problem Statement

A machine learning model trained on a local computer cannot be used by other users.

This project solves that problem by creating a complete pipeline that allows:

1. Training a machine learning model
2. Tracking experiments
3. Saving trained models
4. Serving predictions through an API
5. Automating workflows using GitHub Actions
6. Deploying the application to AWS EC2

---

## Architecture

Dataset
↓
DVC (Data Versioning)
↓
Model Training (Scikit-Learn)
↓
MLflow (Experiment Tracking)
↓
Model Persistence (Joblib)
↓
FastAPI REST API
↓
GitHub Actions (CI/CD)
↓
AWS EC2 Deployment

---

## Technologies Used

### Machine Learning

* Scikit-Learn
* Pandas
* Joblib

### MLOps Tools

* DVC
* MLflow

### API Development

* FastAPI
* Uvicorn

### DevOps & CI/CD

* Git
* GitHub
* GitHub Actions

### Cloud

* AWS EC2

---

## Features

### Data Versioning with DVC

* Tracks dataset changes
* Maintains dataset history
* Supports reproducibility

### Model Training

* Random Forest Classifier
* Automated training pipeline
* Accuracy evaluation

### Experiment Tracking

* MLflow integration
* Parameter logging
* Metric logging
* Experiment management

### Model Persistence

* Saves trained model as model.pkl
* Enables model reuse without retraining

### REST API

* FastAPI-powered prediction service
* Swagger UI documentation
* Real-time prediction endpoint

### CI/CD Automation

* GitHub Actions workflow
* Automatic pipeline execution on code push
* Continuous Integration support

### AWS Deployment

* Hosted on AWS EC2
* Publicly accessible API endpoint

---

## Project Structure

mlops-project/

├── data/
│ ├── dataset.csv
│ └── dataset.csv.dvc

├── models/
│ └── model.pkl

├── src/
│ └── train.py

├── .github/
│ └── workflows/
│ └── train.yml

├── app.py
├── requirements.txt
├── README.md

---

## Workflow

### Step 1: Dataset Management

The dataset is stored and version controlled using DVC.

### Step 2: Model Training

The training script:

* Loads dataset
* Splits train/test data
* Trains Random Forest model
* Calculates accuracy

### Step 3: Experiment Tracking

MLflow tracks:

* Accuracy
* Parameters
* Training runs

### Step 4: Model Saving

The trained model is saved as:

models/model.pkl

### Step 5: API Creation

FastAPI exposes a prediction endpoint:

POST /predict

### Step 6: CI/CD Automation

GitHub Actions automatically:

* Installs dependencies
* Executes training pipeline
* Validates project execution

### Step 7: Cloud Deployment

The application is deployed on AWS EC2 and can be accessed through a public IP.

---

## Installation

Clone Repository

```bash
git clone <repository-url>
cd mlops-project
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate Virtual Environment

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python src/train.py
```

Output:

* Trained model
* Accuracy score
* MLflow logs
* model.pkl

---

## Start MLflow

```bash
mlflow ui
```

Open:

http://localhost:5000

---

## Run FastAPI

```bash
uvicorn app:app --reload
```

Open Swagger UI:

http://localhost:8000/docs

---

## Sample Prediction

Input:

```json
{
  "feature1": 5,
  "feature2": 6
}
```

Output:

```json
{
  "prediction": 1
}
```

---

## AWS Deployment

The application was deployed on AWS EC2.

Deployment steps:

1. Launch EC2 instance
2. Clone repository
3. Install dependencies
4. Start FastAPI application
5. Access prediction endpoint through public IP

---

## What I Learned

Through this project I gained hands-on experience with:

* Machine Learning Operations (MLOps)
* Dataset Versioning using DVC
* Experiment Tracking using MLflow
* Model Serving using FastAPI
* CI/CD using GitHub Actions
* Cloud Deployment using AWS EC2
* End-to-End ML Lifecycle Management

---

## Future Improvements

* Docker Containerization
* AWS S3 integration for DVC remote storage
* MLflow Model Registry
* Automated retraining pipelines
* Model monitoring
* ECS/EKS deployment
* Real-world business datasets

---

## Resume Description

Built and deployed an end-to-end MLOps pipeline using DVC, MLflow, FastAPI, GitHub Actions, and AWS EC2. Implemented dataset versioning, experiment tracking, model serving, CI/CD automation, and cloud deployment for machine learning applications.

---

## Author

Vaishnavi Sharma
