# Gemini Ticket Classification Pipeline

A production-ready ML pipeline that automatically classifies customer support tickets using Google's Gemini AI model, deployed on Google Cloud Platform (GCP) using Vertex AI Pipelines.

## Overview

This pipeline processes customer support ticket data through three main stages:
1. Data Preparation
2. Batch Prediction using Gemini
3. Results Processing and Storage

## Data Source

The pipeline uses the [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset/data) from Kaggle, which includes:
- Customer information (name, email, age, gender)
- Product details
- Purchase dates
- Ticket information (type, subject, description)

## Pipeline Components

### 1. Data Preparation (`prepare_batch_data.py`)
- Reads customer ticket data from Google Cloud Storage
- Extracts key fields:
  - Ticket Subject
  - Ticket Description
- Formats data into Gemini-compatible JSONL format
- Includes prompt template for classification
- Outputs prepared data to GCS for batch processing

### 2. Batch Prediction (`run_batch_prediction.py`)
- Uses Vertex AI's BatchPredictionJob
- Processes tickets using Gemini model
- Classifies tickets into categories:
  - Technical Issue
  - Billing Inquiry
  - Product Inquiry
  - Cancellation Request
  - Refund Request
- Stores prediction results in GCS

### 3. Results Processing (`process_results.py`)
- Retrieves prediction results from GCS
- Combines original ticket data with predictions
- Writes final results to BigQuery table with schema:
  - Original ticket information
  - Predicted category
  - Confidence score
  - Processing timestamp

## Architecture
Raw Data (GCS) → Data Preparation → Batch Prediction → Results Storage (BigQuery)


### Data Transformation Steps

1. **Raw Data → Preparation**
   ```
   CSV in GCS → Extract Ticket Subject/Description → JSONL Format
   ```

2. **Batch Prediction**
   ```
   JSONL → Gemini Model → Classification Results
   ```

3. **Results Processing**
   ```
   Predictions + Original Data → Structured BigQuery Table
   ```

## Data Flow
```
GCS Input Bucket                    ┌──────────────┐
(customer_tickets.csv)              │              │
        │                    ┌──────│ Gemini Model │
        │                    │      │              │
        ▼                    │      └──────┬───────┘
   Data Prep                 │             │
   Component                 │             │
        │                    │             │
        ▼                    │             │
GCS Batch Storage            │             ▼
(batch_input.jsonl) ─────────┘      GCS Output Bucket
                                  (predictions/*.jsonl)
                                            │
                                            ▼
                                    Results Processor
                                            │
                                            ▼
                              BigQuery Classification Table
```

### Step-by-Step Flow:

1. **Input Storage (GCS)**
   - Raw customer support ticket CSV
   - Contains customer data, ticket details

2. **Data Preparation**
   - Reads CSV from GCS
   - Extracts ticket subject and description
   - Creates JSONL format for batch processing
   - Stores in GCS as batch_input.jsonl

3. **Batch Prediction**
   - Gemini model processes tickets
   - Classifies into support categories
   - Stores results in GCS output bucket

4. **Results Processing**
   - Combines original data with predictions
   - Writes final structured data to BigQuery
