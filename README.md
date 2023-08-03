# NLP MCTI LDA Recommender 📊

---
title: NLP MCTI LDA Recommender
emoji: 📊
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 3.9.1
app_file: app.py
pinned: false
---

This repository contains the code for a Gradio app that implements a unique recommendation system using the Netflix svd++ algorithm. The app is hosted on Hugging Face at the following URL: [NLP MCTI LDA Recommender](https://huggingface.co/spaces/unb-lamfo-nlp-mcti/nlp-mcti-lda-recommender)


## About the App

The NLP MCTI LDA Recommender is a unique application that simulates a recommendation system. It uses the Netflix svd++ algorithm, a popular machine learning model for recommendation systems, to generate recommendations. 

However, unlike traditional recommendation systems, this app doesn't rely on a pre-existing user dataset for training. Instead, it simulates users by training a Latent Dirichlet Allocation (LDA) model on the text of opportunities. The latent topics derived from the LDA model are interpreted as potential users.

The simulated user classification dataset is generated based on the proportion of each topic present in each opportunity text. For example, if a text (let's say text 32) had 80% of topic 2, then the simulated "user 2" would give 5 stars to this opportunity. The star rating decreases as the proportion of the topic in the text decreases:

- 80% to 100% of the topic: 5 stars
- 60% to 80% of the topic: 4 stars
- 40% to 60% of the topic: 3 stars
- 20% to 40% of the topic: 1 star
- 0% to 20% of the topic: 0 stars

This innovative approach allows the app to generate  simulated recommendations based on the contents of each opportunity even when a traditional user dataset is not available, making it a versatile tool for a variety of recommendation scenarios.

