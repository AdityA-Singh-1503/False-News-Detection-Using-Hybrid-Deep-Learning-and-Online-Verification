# Problem Statement

False news spreads across social media incredibly fast these days. People constantly share wild headlines without double-checking them, and it ends up causing a lot of real-world mess.

The core issue I wanted to tackle is that standard machine learning models are pretty bad at catching brand-new rumors—they only know what they were historically trained on. But at the same time, expecting humans to manually fact-check every single post is painfully slow and impossible to scale.

To get around this, I decided to build a hybrid detection system. The goal was to combine the best of both worlds: the system uses a fine-tuned deep learning model to analyze the text locally, but then it immediately hits the internet to cross-check the claim against live, trusted sources. Basically, it pairs the instant categorization of an offline AI with the hard proof of a real-time web search.

# Objective

The main goals for this project basically boiled down to this:

1. Build an ML classifier that can accurately flag an article as legit or fabricated.
2. Under the hood, fine-tune a DistilBERT transformer using the WELFake data.
3. Run the standard metrics (accuracy, precision, recall, and F1-score) to see how well it's really performing.
4. Hook the system into the live internet for real-time verification via:

   * Wikipedia API
   * GNews API
   * Google Fact Check Tools API

5. Build a hybrid prediction system that combines AI classification with real-time fact verification.
6. Provide a Streamlit-based user interface for end-users to test and verify news.

# Scope of the Project

As for the scope, this tool is built specifically to analyze English text—so it doesn't handle images, video, or audio misinformation. It also relies heavily on third-party APIs to pull in live facts. While that live-checking makes the system way more accurate, it does mean the tool is constrained by the request limits of those free-tier APIs.

# Expected Outcomes

At the end of the day, here is what came out of this project:

A final product that is way more trustworthy than your standard, isolated machine learning classifier.
1. A trained DistilBERT model capable of classifying news with high accuracy.
2. A hybrid verification system that cross-checks news authenticity using online sources.
3. A user-friendly Streamlit application that allows real-time testing.
4. A performance report including accuracy, precision, recall, F1-score, and confusion matrix.
5. Improved reliability in detecting fake news compared to a standalone machine learning model.
