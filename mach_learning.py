'''
A successful ML model must be accompanied with several things:
* Error analysis on training data and testing data
* Predictions on new or hypothetical data. Ideally these can be plotted.
* Plots of predictions that illustrate how predictions are made on test data
* feature importance analysis that provide insight on what features impact predictions the most

Plotting should be done in main.py.
'''
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle #for saving model on pc

#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

def model_regression(features, labels, test_size=0.3, validation_split = 0.5): #validation split not required for regression
    print("starting model training")
    # Split the data into training and testing sets
    train_f, test_f, train_l, test_l = train_test_split(features, labels, test_size=test_size)

    #we further split test_f and test_l so we have a validation set, so we can tune the model
    #test_f, val_f, test_l, val_l = train_test_split(test_f, test_l, test_size=validation_split)

    # Initialize the Linear Regression model & train it
    #loss = {'modified_huber', 'squared_hinge', 'log_loss', 'squared_error', 'huber', 'perceptron', 'epsilon_insensitive', 'squared_epsilon_insensitive', 'hinge'}
    model = SGDClassifier(loss='log_loss', max_iter=100000, random_state=42)
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

#Our data has 5 fundemental catagories of information about crashes, our whole project
#is about showing how these 4 different factors contribute to the outcomes of a crash
#when = ['Start_Time','Sunrise_Sunset'] #overall hour plot, per state hour plot
#where = ['Start_Lat','Start_Lat','State'] #states graph, and united states map DONE
#weather = ['Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Weather_Condition'] #visibility interactive graph
#road_elements = ['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop'] #graph showing 10 most common road elements
#-------------------------------------------------------#
#outcomes = ['Distance(mi)','Severity','End_Time','Start_Time'] #End_Time-Start_Time = Duration, Data existing at all means it was crash
def prep_data(df):
    #weather condition has a lot of dummies
    essentialcolumns = ['Severity','Start_Time','Sunrise_Sunset','Start_Lat','Start_Lat',
                        'State','Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)',
                        'Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Weather_Condition',
                        'Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout',
                        'Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']

    df = df.loc[:,essentialcolumns]
    df = df.dropna()
    labels = df["Severity"].copy()
    df = df.drop('Severity', axis=1)
    #fixing time ---
    df["Start_Time"] = df["Start_Time"].apply(lambda x : str(str(x)[0:19]))
    df["Start_Time"] = pd.to_datetime(df['Start_Time'])

    df["Hour"] = df['Start_Time'].dt.hour
    df["Month"] = df['Start_Time'].dt.month
    df["Day"] = df['Start_Time'].dt.day
    df = df.drop("Start_Time", axis=1)
    # Weather_Condition is categorical
    df = pd.get_dummies(df,columns=["Sunrise_Sunset"]) #out of memory error, going to hope that it is unescessary
    df = pd.get_dummies(df,columns=["State"])
    df = pd.get_dummies(df,columns=["Weather_Condition"])
    print("finished prepping ml data")
    return df, labels

if __name__ == "__main__":
    model = None
    feat,label = prep_data(pd.concat(
        map(pd.read_csv, ['data\output_0.csv', 'data\output_1.csv', 'data\output_2.csv', 'data\output_3.csv', 'data\output_4.csv', 'data\output_5.csv', 'data\output_6.csv', 'data\output_7.csv'])
        ))
    if False:
        model = model_regression(feat,label)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        with open('model.pkl', 'rb') as f: #load model back into memory
            model = pickle.load(f)
    #show_coefficients(model,feat)
    show_importance(model, feat)