# Gender Detection Using Machine Learning and Flask

## Project Overview

This project predicts the gender of a person based on their name using Machine Learning. The user enters a name through a Flask web application, and the trained classification model predicts whether the name is Male or Female along with a confidence score.

## Technologies Used

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* HTML
* CSS

## Machine Learning Algorithm

* Multinomial Naive Bayes Classification

## Project Structure

GenderDetection/

├── app.py

├── train_model.py

├── gender_dataset.csv

├── model.pkl

├── vectorizer.pkl

├── templates/

│ └── index.html

├── static/

│ └── style.css

└── README.md

## Features

* Name-based gender prediction
* Confidence score display
* User-friendly web interface
* Machine Learning classification model
* Fast prediction using saved model files

## How to Run

### Install Required Packages

pip install flask pandas numpy scikit-learn

### Train the Model

python train_model.py

### Run the Application

python app.py

### Open Browser

http://127.0.0.1:5000

## Sample Output

Input Name: Priya

Predicted Gender: Female

Confidence Score: 94%

## Future Enhancements

* Larger dataset for better accuracy
* CSV file upload support
* Prediction history
* User authentication
* Graphical analytics dashboard

## Author

Developed as a Machine Learning Mini Project using Flask and Scikit-learn.
