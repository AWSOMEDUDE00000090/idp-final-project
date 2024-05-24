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

#RandomForestClassifier or GradientBoostingClassifier for importance

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

def model_mean_square(model,features, labels, test_size=0.3):
    # Split the data into training and testing sets
    train_f, test_f, train_l, test_l = train_test_split(features, labels, test_size=test_size)
    # Test the accuracy: Make predictions and show our MSE
    label_predictions = model.predict(test_f)
    print(f'MSE : {mean_squared_error(test_l, label_predictions):.2f}')

def show_coefficients(model, features):
    # Now, get the important of each feature
    # Get the coefficients and feature names
    coefficients = model.coef_
    feature_names = features.columns
    print("features for coeficients size" + str(len(list(feature_names))))
    print(list(feature_names))
    #print("coef:")
    #print(model.coef_)
    # Print the y-intercept
    sum = 0
    for i in model.coef_:
        for j in i:
            sum += 1
    print('total size = ' + str(sum))
    for i in model.intercept_:
        #sgdclassifier has multiple intercepts apperantly
        print(f'Intercept: {i:.0f}')
    # Print the coefficients and feature names
    '''
    When there are more coefficients than columns, it means that the model is using additional coefficients to capture the relationships between the features and the target variable
    '''
    for f, c in zip(feature_names, coefficients):
        #print(f'{f}   : {c:.3f}')
        print("-------")
        print(f)
        print("-------")
        print(c)
        print("-------")

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
    essentialcolumns = ['Severity','Start_Time','Sunrise_Sunset','Start_Lat','Start_Lng',
                        'State','Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)',
                        'Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Weather_Condition',
                        'Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout',
                        'Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']

    df = df.loc[:,essentialcolumns]
    #print(df.info())
    #print("------")
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
    #print(list(df.columns))
    print("finished prepping ml data")
    return df, labels

def create_all_features(df):
    '''
    ['Severity','Start_Time','Sunrise_Sunset','Start_Lat','Start_Lng',
    'State','Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)',
    'Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Weather_Condition',
    'Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout',
    'Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']
    '''
    #find ranges by doing max and min on all of the columns
    bool_cols = df.select_dtypes(include=[bool]).columns
    num_cols = pd.DataFrame()
    num_cols["Name"] = df.select_dtypes(include=[int, float]).columns

    boolean_variations = df.loc[:,bool_cols].unique()
    print(boolean_variations)

    num_cols["low"] = num_cols["name"]
    # lets create a set of data to make predictions on
    # get all ages for each sport and gender
    sports = df['sport'].unique()
    ages = [ age for age in range(18, 101) ]
    genders = [False] * (len(ages)*len(sports))
    gender_m = [True] * len(genders)
    genders.extend(gender_m)
    # replicate ages to be for all sports (multiply by number of sports)
    # then, double itself to get both genders
    all_ages = []
    for i in range(len(sports)):
        all_ages.extend(ages)
    all_ages.extend(all_ages)
    # replicate sports to be for all ages (multiply by number of ages)
    # then, double itself to get both genders
    all_sports = []
    for i in range(len(ages)):
        all_sports.extend(sports)
    all_sports.extend(all_sports)

    # create the dataframe structured like original features during training
    df_features = pd.DataFrame({'male':genders, 'age':all_ages, 'sport':all_sports})
    return df_features

if __name__ == "__main__":
    model = None
    newmodel = False
    feat,label = prep_data(pd.concat(
        map(pd.read_csv, ['data\output_0.csv', 'data\output_1.csv', 'data\output_2.csv', 'data\output_3.csv', 'data\output_4.csv', 'data\output_5.csv', 'data\output_6.csv', 'data\output_7.csv'])
        ))
    if newmodel:
        model = model_regression(feat,label)
        with open('model2.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        with open('model2.pkl', 'rb') as f: #load model back into memory
            model = pickle.load(f)
    model_mean_square(model,feat,label)
    show_coefficients(model,feat)
    #show_importance(model, feat)