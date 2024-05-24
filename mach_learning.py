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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,mean_squared_error
import pandas as pd
import pickle #for saving model on pc
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

#RandomForestClassifier or GradientBoostingClassifier for importance

def model_SGD(features, labels, test_size=0.3, validation_split = 0.5): #validation split not required for regression
    print("starting model training")
    # Split the data into training and testing sets
    train_f, test_f, train_l, test_l = train_test_split(features, labels, test_size=test_size)

    #we further split test_f and test_l so we have a validation set, so we can tune the model
    #test_f, val_f, test_l, val_l = train_test_split(test_f, test_l, test_size=validation_split)

    # Initialize the Linear Regression model & train it
    #loss = {'modified_huber', 'squared_hinge', 'log_loss', 'squared_error', 'huber', 'perceptron', 'epsilon_insensitive', 'squared_epsilon_insensitive', 'hinge'}
    model = SGDClassifier(loss='log_loss', max_iter=100000, random_state=42)
    #sgd optimizaition for logistic regression
    model.fit(train_f, train_l)

    # Test the accuracy: Make predictions and show our MSE
    label_predictions = model.predict(test_f)
    print(f'MSE : {mean_squared_error(test_l, label_predictions):.2f}')
    
    return model

def model_RFC(features, labels, test_size=0.3, validation_split = 0.5): #validation split not required for regression
    print("starting model training")
    # Split the data into training and testing sets
    train_f, test_f, train_l, test_l = train_test_split(features, labels, test_size=test_size)

    #we further split test_f and test_l so we have a validation set, so we can tune the model
    #test_f, val_f, test_l, val_l = train_test_split(test_f, test_l, test_size=validation_split)

    # Initialize the Linear Regression model & train it
    #loss = {'modified_huber', 'squared_hinge', 'log_loss', 'squared_error', 'huber', 'perceptron', 'epsilon_insensitive', 'squared_epsilon_insensitive', 'hinge'}
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    #sgd optimizaition for logistic regression
    model.fit(train_f, train_l)

    # Test the accuracy: Make predictions and show our MSE
    label_predictions = model.predict(test_f)
    print(f'MSE : {mean_squared_error(test_l, label_predictions):.2f}')
    
    #print("Accuracy:", accuracy_score(test_f, label_predictions))
    #print("Classification Report:")
    #print(classification_report(test_f, label_predictions))
    #print("Confusion Matrix:")
    #print(confusion_matrix(test_f, label_predictions))

    return model

def model_SGD(features, labels, test_size=0.3, validation_split = 0.5): #validation split not required for regression
    print("starting model training")
    # Split the data into training and testing sets
    train_f, test_f, train_l, test_l = train_test_split(features, labels, test_size=test_size)

    #we further split test_f and test_l so we have a validation set, so we can tune the model
    #test_f, val_f, test_l, val_l = train_test_split(test_f, test_l, test_size=validation_split)

    # Initialize the Linear Regression model & train it
    #loss = {'modified_huber', 'squared_hinge', 'log_loss', 'squared_error', 'huber', 'perceptron', 'epsilon_insensitive', 'squared_epsilon_insensitive', 'hinge'}
    model = SGDClassifier(loss='log_loss', max_iter=100000, random_state=42)
    #sgd optimizaition for logistic regression
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

def show_coefficients_SGD(model, features):
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

def show_importance(model3, features):
    # get importance
    importance = model3.feature_importances_
    # summarize feature importance
    feat = []
    impor = []
    feat2 = []
    impor2 = []
    total_poi = 0
    total_weather = 0
    total_state = 0
    for index, feat_importance in enumerate(importance):
        print(f'Feature: {features.columns[index]}, Importance: {feat_importance:.2%}')
        
        if feat_importance < 0.02:
            feat2.append(features.columns[index])
            impor2.append(feat_importance)
            if features.columns[index].startswith("Weather"):
                total_weather += feat_importance
            elif features.columns[index].startswith("State"):
                total_state += feat_importance
            else:
                total_poi += feat_importance
        else:
            feat.append(features.columns[index])
            impor.append(feat_importance)
    feat.append("Road Element")
    impor.append(total_poi)
    feat.append("Weather")
    impor.append(total_weather)
    feat.append("State")
    impor.append(total_state)
    fig, ax = plt.subplots()
    ax.pie(impor, labels=feat, autopct='%1.1f%%')#,textprops={'fontsize': 5}
    plt.title('Feature importance in RFC, used to predict Severity of Accidents')
    plt.savefig('mlfeatureimportance.jpg', dpi=600)
    plt.close(fig)

    '''
    #the <2% graph is inconsequential and not important
    #I tried piechart, and bargraph before this
    fig, ax = plt.subplots()
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature <2% importance in RandomForest Classifier")
    plt.scatter(impor2,range(len(feat2)))
    outliers = [(i, v) for i, v in zip(impor2,range(len(feat2)))]
    outliertext = []
    for x, y in outliers:
        outliertext.append(plt.annotate(f'{feat2[y]}: {x:.1e}%', xy=(x, y)))
    adjust_text(outliertext, arrowprops=dict(arrowstyle='->', color='red'))
    #ax.bar(range(len(impor2)),impor2, label=feat2)
    plt.title('Feature <2% importance in RandomForest Classifier')
    plt.savefig('mlfeatureimportance_small.jpg', dpi=600)
    plt.close(fig)
    '''
    

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

if __name__ == "__main__":
    model = None
    model2 = None
    #generate new model using True, otherwise leave false to read stored model
    newmodel = False
    newmodel2 = False
    feat,label = prep_data(pd.concat(
        map(pd.read_csv, ['data\output_0.csv', 'data\output_1.csv', 'data\output_2.csv', 'data\output_3.csv', 'data\output_4.csv', 'data\output_5.csv', 'data\output_6.csv', 'data\output_7.csv'])
        ))
    if newmodel:
        model = model_SGD(feat,label)
        with open('model2.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        with open('model2.pkl', 'rb') as f: #load model back into memory
            model = pickle.load(f)
            '''
            InconsistentVersionWarning: Trying to unpickle estimator SGDClassifier from version 1.3.1 
            when using version 1.5.0. This might lead to breaking code or invalid results.
            Use at your own risk. For more info please refer to: 
            https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
            '''
    #model_mean_square(model,feat,label)
    #show_coefficients_SGD(model,feat)
    #show_importance(model, feat)
    if newmodel2:
        model2 = model_RFC(feat,label)
        with open('model_rfc.pkl', 'wb') as f:
            pickle.dump(model2, f)
    else:
        with open('model_rfc.pkl', 'rb') as f: #load model back into memory
            model2 = pickle.load(f)
    show_importance(model2,feat)