# HOTS-Classifier

The Text Classification project aims to classify text into three classes: HOTS (Highly Occurring Text Sequences), MOTS (Moderately Occurring Text Sequences), and LOTS (Low Occurring Text Sequences). The goal is to accurately categorize text based on their occurrence frequency. The project utilizes two classification algorithms: Decision Tree Classifier and K-Nearest Neighbors (KNN), and applies TF-IDF (Term Frequency-Inverse Document Frequency) as a preprocessing technique.

## Features
- Classify text into three categories: HOTS, MOTS, and LOTS based on their occurrence frequency.
- Utilize Decision Tree Classifier and KNN algorithms for classification.
- Apply TF-IDF as a preprocessing technique to represent text data.
- Explore hyperparameter tuning to optimize the classification models.

## Model Development

The project involves the development of two classification models: Decision Tree Classifier and KNN.

- Decision Tree Classifier: This model builds a decision tree based on the features derived from the TF-IDF representation of the text data. It splits the data based on the occurrence frequency of text sequences to classify them into the respective classes.

- K-Nearest Neighbors (KNN): This model utilizes the TF-IDF representation to measure the similarity between the input text and the training instances. It classifies the text by considering the k nearest neighbors in the training data.

Both models undergo hyperparameter exploration to find the optimal values for parameters such as maximum depth, criterion, number of neighbors, and distance metric.


## Installation

To install and run the project, follow these steps:

1. Clone this repository to your local machine.
2. Install the necessary dependencies using pip. You can do this by running the following command in your terminal:
`pip install -r requirements.txt`
3. Run the web app using the following command: `streamlit run app.py`
4. Access the web app by opening your web browser and navigating to `http://localhost:8501`.

## Usage

Just input your text into the app and it'll classify the data for you.

## Example

Here's an example of how to use the web app:

1. Open your web browser and navigate to `http://localhost:8501`.
2. Select a sample text from the provided options or enter your own text in the input field.
3. Click the "Predict" button to classify the text into HOTS, MOTS, or LOTS.

## Issues

There issues related to storages on this project.

## Acknowledgements

This project was built using scikit-learn and Streamlit framework.
