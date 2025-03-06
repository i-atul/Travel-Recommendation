# Travel-Recommendation

Get personalized travel recommendations based on user preferences and collaborative filtering.

## Project Overview

This project provides a travel recommendation system that suggests destinations based on user preferences and historical data. It uses machine learning models to predict the popularity of destinations and collaborative filtering to recommend destinations based on user similarity.

## Features

- Personalized travel recommendations
- Collaborative filtering based on user similarity
- Popularity prediction of destinations


## Setup Instructions

1. Clone the repository:
    ```sh
    git clone https://github.com/i-atul/Travel-Recommendation.git
    cd Travel-Recommendation
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```


## Usage


### Training the Model

If you need to retrain the model, run the `train_model.py` script:
```sh
python train_model.py
```

### Running the Application

1. Start the Flask application:
    ```sh
    python app.py
    ```

2. Open your web browser and navigate to `http://localhost:5010`.


### Making Recommendations

1. Navigate to the home page and enter the required user information.
2. Submit the form to get personalized travel recommendations and predicted popularity scores.

## Project Structure

- `app.py`: Main Flask application file.
- `train_model.py`: Script to train the machine learning model.
- `Notebook/`: Directory containing datasets and trained models.
- `templates/`: HTML templates for the web application.



## License

This project is licensed under the MIT License.
