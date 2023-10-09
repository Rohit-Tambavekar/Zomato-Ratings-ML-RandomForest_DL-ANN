# Zomato-Ratings-ML-RandomForest_DL-ANN
Welcome to the Zomato Ratings project! This project is a Streamlit web application that predicts restaurant ratings based on user inputs. We have used Random Forest and Artificial Neural Networks (ANN) to calculate the rating for the provided input parameters.

You can access the live application here: [Zomato Ratings App](https://zomato-ratings.streamlit.app/)

## Table of Contents
- [Project Files](#project-files)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Models](#models)
- [Data](#data)
- [Contributing](#contributing)
- [License](#license)

## Project Files

The project repository includes the following files and directories:

- `.streamlit`: Configuration folder for Streamlit.
- `.gitattributes`: Git configuration file.
- `.gitignore`: Git ignore rules.
- `DockerFile`: Dockerfile for containerization.
- `LICENSE`: License file.
- `README.md`: This README file.
- `ann_model.h5`: Trained Artificial Neural Network model in HDF5 format.
- `ann_model.pkl`: Pickle file containing the ANN model.
- `label_encoder.pkl`: Pickle file for label encoding.
- `main.py`: The main Streamlit application code.
- `model_fit.py`: Code for model training.
- `multi_label_binarizer.pkl`: Pickle file for multi-label binarizer.
- `online_order_mapping.pkl`: Pickle file for online order mapping.
- `random_forest_model.pkl.bz2`: Compressed pickle file for the Random Forest model.
- `requirements.txt`: List of Python dependencies.
- `sc_model.pkl`: Pickle file for the scikit-learn model.
- `zomato_test.csv`: Test data for the model.
- `zomato_train.csv`: Training data for the model.

## Getting Started

To get started with this project, you can follow these steps:

1. Clone this repository to your local machine.

```
git clone https://github.com/Rohit-Tambavekar/Zomato-Ratings-ML-RandomForest_DL-ANN.git
```

2. Install the required dependencies by running:

```
pip install -r requirements.txt
```

3. Run the Streamlit application:

```
streamlit run main.py
```

This will launch the Zomato Ratings web app in your browser, where you can input restaurant details and get the predicted rating.

## Usage

In the Streamlit app, you can provide the following input parameters to get a restaurant rating prediction:

- Online Orders (Yes/No)
- Table Booking (Yes/No)
- Location
- Restaurant Type
- Cuisines
- Restaurant Listed In Type
- Cost for Two
- Votes

After entering the required details, click the "Predict" button to see the predicted rating.

## Dependencies

The project relies on the following Python libraries and packages listed in `requirements.txt`:
```
pandas
numpy
streamlit
streamlit-option-menu
scikit-learn
tensorflow
```

You can install these dependencies using the provided `requirements.txt` file.

## Models

We have employed both Random Forest and Artificial Neural Network (ANN) models to predict restaurant ratings. The trained models are available as `random_forest_model.pkl.bz2` and `ann_model.h5` in the project directory.

## Data

The project utilizes training and test data, available as `zomato_train.csv` and `zomato_test.csv`, respectively. These datasets were used to train and evaluate the models.

## Contributing

Contributions to this project are welcome. Feel free to submit bug reports, feature requests, or pull requests to help improve the project.

## License

This project is licensed under the [MIT License](LICENSE).

Thank you for using Zomato Ratings! If you have any questions or feedback, please don't hesitate to reach out. Enjoy predicting restaurant ratings with our app!
