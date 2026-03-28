# False News Detector using Hybrid Deep Learning and Online Verification

This project is a hybrid approach to catching fake news. Instead of just relying on machine learning, it combines a fine-tuned DistilBERT model with live fact-checking. The model predicts if a claim looks authentic, and then the system immediately verifies that prediction against live data from Wikipedia, GNews, and Google Fact Check.
The project covers everything from training the model to writing the final prediction logic, and I wrapped it all up in a Streamlit web interface to make hands-on testing super easy.

## 1. Project Overview

Detecting false news is notoriously tricky since AI models often stumble when evaluating unfamiliar or unseen news. To solve this, the project relies on a hybrid setup:

1. An offline, fine-tuned DistilBERT model handles the baseline text classification.
2. The system then cross-references the user's input with live online verification using:
  
   * GNews live articles
   * Google Fact Check API
   * Wikipedia summary

(Use your own API KEY)

The hybrid logic combines all three sources to produce a more reliable final verdict.

## 2. Dataset Used

The model is trained using the **WELFake Dataset**, which consists of both real and fake news articles.

* Total dataset size: 70,000+ records
* Includes title, text, and label
* Label 1 = Fake, Label 0 = Real

The dataset was split into training and testing using an 80:20 ratio.

## 3. Model Used

The project uses:

* DistilBERT (distilbert-base-uncased)
* Fine-tuned for binary classification
* Training was done for one epoch with AdamW optimizer and learning rate of 2e-5

The trained model is saved in `models/best_model/`.

## 4. Evaluation Results

The model was evaluated on 14,427 test samples.

**Results:**

* Accuracy: 97.93%
* Precision: 95.76%
* Recall: 99.04%
* F1-Score: 98.83%

**Confusion Matrix:**

|             | Predicted Real | Predicted Fake |
| ----------- | -------------- | -------------- |
| Actual Real | 6965           | 96             |
| Actual Fake | 64             | 7274           |

The model performs strongly on both classes with minimal false positives and false negatives.

---

## 5. Hybrid Verification System

Since AI models alone aren't perfect—especially when a story just broke—the project uses a hybrid approach to double-check everything.

When a user drops in a piece of news:

1. The offline AI steps in first, predicting if it's FAKE or REAL along with a confidence score.
2. Right after that, the system runs a live web check to verify the facts through:

   * GNews API for recent news articles
   * Google Fact Check Tools API
   * Wikipedia API search

Hybrid logic:

* If any fact-check source marks the claim as false → Final result is FAKE.
* If Wikipedia strongly matches the topic → Mark as TRUE.
* If live news articles support the claim → Mark as TRUE.
* If offline model says fake but no evidence online → Mark as UNCERTAIN.
* Otherwise, fallback to the AI prediction.

This hybrid approach significantly increases overall reliability.

## 6. Folder Structure

FakeNews-main/
│
├── data/
│   ├── WELFake_Dataset.csv
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── best_model/
│
├── dataset.py
├── train_distilbert.py
├── evaluate_model.py
├── local_test.py
├── fact_check.py
├── online_sources.py
├── hybrid_predict.py
├── online_verify.py
├── preprocess.py
├── streamlit_app.py
├── make_test_split.py
├── utility.py
└── test_api.py

## 7. How to Run the Project

### Step 1: Create and activate a virtual environment

Windows:

python -m venv venv
.\venv\Scripts\activate

### Step 2: Install required libraries

pip install -r requirements.txt

### Step 3: Train the model (optional)

python train_distilbert.py

### Step 4: Evaluate the model

python make_test_split.py
python evaluate_model.py

### Step 5: Run the hybrid verification system

python online_verify.py

### Step 6: Run the Streamlit app

streamlit run streamlit_app.py

## 8. Features Implemented

* Put together a simple Streamlit UI so anyone can test it out easily.
* Built a hybrid verification step that actually pings three different live APIs.
* Handled all the boring data prep—merging, text cleaning, and setting up the train/test splits.
* Fine-tuned a DistilBERT model specifically for classifying the text.
* Evaluated the model using the usual metrics (accuracy, precision, recall, F1) to see how well it actually works.
* Kept all the code modular and clean so it's easy to read and tweak later

## 9. Limitations

* Offline model accuracy depends on dataset quality.
* Wikipedia and GNews can sometimes return unrelated pages.
* Free API rate limits may restrict the number of online checks per day.
* Hybrid system relies on internet connectivity.

## 10. Future Improvements

* Upgrading the main model from DistilBERT to something stronger, like DeBERTa-v3.
* Improving how the system searches Wikipedia by adding sentence transformers for better context matching.
* Integrating a wider variety of news APIs to strengthen the fact-checking side of things.
* Polishing up the Streamlit interface to make it look and feel a lot better.
* Pushing the final app live to the web.

