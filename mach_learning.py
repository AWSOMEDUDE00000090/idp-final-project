'''
A successful ML model must be accompanied with several things:
* Error analysis on training data and testing data
* Predictions on new or hypothetical data. Ideally these can be plotted.
* Plots of predictions that illustrate how predictions are made on test data
* feature importance analysis that provide insight on what features impact predictions the most

Plotting should be done in main.py.
'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

def model_regression(features, labels, test_size=0.3, validation_split = 0.5): #validation split not required for regression
    print("starting model training")
    # Split the data into training and testing sets
    train_f, test_f, train_l, test_l = train_test_split(features, labels, test_size=test_size)

    #we further split test_f and test_l so we have a validation set, so we can tune the model
    #test_f, val_f, test_l, val_l = train_test_split(test_f, test_l, test_size=validation_split)

    # Initialize the Linear Regression model & train it
    model = LinearRegression()
    model.fit(train_f, train_l)

    # Test the accuracy: Make predictions and show our MSE
    label_predictions = model.predict(test_f)
    print(f'MSE : {mean_squared_error(test_l, label_predictions):.2f}')
    
    return model

def show_coefficients(model, features):
    # Now, get the important of each feature
    # Get the coefficients and feature names
    coefficients = model.coef_
    feature_names = features.columns

    # Print the y-intercept
    print(f'Intercept: {model.intercept_:.0f}')
    # Print the coefficients and feature names
    for f, c in zip(feature_names, coefficients):
        print(f'{f}   : {c:.3f}')

def show_importance(model, features):
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for index, feat_importance in enumerate(importance):
        print(f'Feature: {features.columns[index]}, Importance: {feat_importance:.2%}')

def prep_data(df):
    df = df[["ID","Start_Time","Start_Lat","Start_Lng","Temperature(F)","Wind_Chill(F)","Humidity(%)",
             "Pressure(in)","Visibility(mi)","Wind_Direction","Wind_Speed(mph)","Precipitation(in)","Weather_Condition","Amenity",
             "Bump","Crossing","Give_Way","Junction","No_Exit","Railway","Roundabout","Station","Stop","Traffic_Calming","Traffic_Signal",
             "Turning_Loop","Sunrise_Sunset","Civil_Twilight","Nautical_Twilight","Astronomical_Twilight","Severity"
             ]]
    df = df.dropna()
    labels = df["Severity"]
    features = df.drop('Severity', axis=1)
    # Weather_Condition is categorical
    #features = pd.get_dummies(features) #out of memory error, going to hope that it is unescessary
    print("finished prepping ml data")
    return features, labels