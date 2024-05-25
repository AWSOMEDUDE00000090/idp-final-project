'''
A successful ML model must be accompanied with several things:
* Error analysis on training data and testing data
* Predictions on new or hypothetical data. Ideally these can be plotted.
* Plots of predictions that illustrate how predictions are made on test data
* feature importance analysis that provide insight on what features impact predictions the most

Plotting should be done in main.py.
'''
#from sklearn.linear_model import LinearRegression
import time
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,mean_squared_error
import pandas as pd
import pickle #for saving model on pc
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay


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

def model_RFC(features, labels, test_size=0.3, validation_split=0.5, model=None):
    print("starting model training")
    
    # split the data 
    train_f, test_f, train_l, test_l = train_test_split(features, labels, test_size=test_size)

    # initialize the random forest classifier
    if model is None:
        model = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        best_score = float('-inf')
        best_params = None

        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                for min_samples_split in param_grid['min_samples_split']:
                    for min_samples_leaf in param_grid['min_samples_leaf']:
                        model.set_params(n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf)

                        # Train the model
                        print(n_estimators,max_depth,min_samples_split,min_samples_leaf)
                        model.fit(train_f, train_l)

                        # Evaluate the model
                        label_predictions = model.predict(test_f)
                        mse = mean_squared_error(test_l, label_predictions)

                        # Update best hyperparameters if current configuration is better
                        if mse > best_score:
                            best_score = mse
                            best_params = {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf
                            }

        model.set_params(**best_params)
        print("Best hyperparameters found:", best_params)

    else:
        model.fit(train_f, train_l)

    label_predictions = model.predict(test_f)
    print(f'MSE: {mean_squared_error(test_l, label_predictions):.2f}')

    # plot partial dependence plots
    bool_cols = features.select_dtypes(include=[bool]).columns
    num_cols = features.select_dtypes(include=[int, float]).columns
    plot_ml_partial_dependence(model, train_f, num_cols, bool_cols)
    plot_ml_partial_dependence(model, train_f, num_cols, bool_cols, target=4)
    
    return model

#https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html#way-partial-dependence-with-different-models
#I tried writing my own in create_all_features explanation on why it didn't work is there
def plot_ml_partial_dependence(model,X_train,features,categorical_data,target=1):
    common_params = {
        "subsample": 50,
        "n_jobs": 2,
        "grid_resolution": 20,
        "random_state": 0,
    }

    print("Computing partial dependence plots...")
    features_info = {
        # features of interest
        "features": features,
        # type of partial dependence plot
        "kind": "average",
        # information regarding categorical features
        "categorical_features": categorical_data,
    }
    tic = time.perf_counter()
    _, ax = plt.subplots(ncols=3, nrows=4, figsize=(9, 8), constrained_layout=True)
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        **features_info,
        ax=ax,
        **common_params,
        target=target
    )
    print(f"done in {time.perf_counter() - tic:.3f}s")
    _ = display.figure_.suptitle(
        (
            "Partial dependence of all numerical features for the car crash \n"
            "dataset with an RandomForestClassifier targeting severity "+str(target)
        ),
        fontsize=16,
    )
    plt.savefig('mlpartialdependency'+str(target)+'.jpg', dpi=600)


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
    return df, labels, df.columns

def create_all_features(df,label,model,featcols):
    #this is the wrong approach, well kindof.
    #without googling i made this method, and then afterwards found out
    #its remarkably similar to how partial dependency graphs work
    #now the cool thing is that with tree, you need to manipulate at least 2 variables to see any changes
    #meaning this method wont get interesting results

    #df = feat.merge(label,how = 'inner', right_index = True, left_index = True)#should be on index
    print("create all features start")
    '''
    ['Severity','Start_Time','Sunrise_Sunset','Start_Lat','Start_Lng',
    'State','Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)',
    'Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Weather_Condition',
    'Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout',
    'Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']
    '''
    #TODO find ranges by doing max and min on all of the columns
    bool_cols = df.select_dtypes(include=[bool]).columns
    num_cols = pd.DataFrame()
    num_cols["Name"] = df.select_dtypes(include=[int, float]).columns
    #TODO get constant averages for all values
    
    num_cols['min'] = [df[i].min() for i in num_cols["Name"]]
    num_cols['max'] = [df[i].max() for i in num_cols["Name"]]
    num_cols['avg'] = [df[i].mean() for i in num_cols["Name"]] #location mean?

    names = list(num_cols["Name"])

    num_cols.set_index('Name', inplace=True)

    weather = list()
    states = list()
    poi = list()
    for i in range(len(bool_cols)):
        if bool_cols[i].startswith("Weather"):
                weather.append(bool_cols[i])
        elif bool_cols[i].startswith("State"):
            states.append(bool_cols[i])
        else:
            poi.append(bool_cols[i])
    #no need for bool cols anymore

    #what we want is columns like this
    # a b c d e f g h     i
    # 1 6 7 8 7 9 3 false false
    # 2 6 7 8 7 9 3 false false
    # 3 6 7 8 7 9 3 false false
    # 4 6 7 8 7 9 3 false false
    for i in names:
        final = pd.DataFrame()
        max = num_cols.loc[i,"max"]
        min = num_cols.loc[i,"min"]
        ml_data = list()
        graph_data = list()
        '''
        #how to fix the np.arrange floating point imprecision
        start = 0.0
        stop = 0.6
        step = 0.2

        num = round((stop - start) / step) + 1   # i.e. length of resulting array
        np.linspace(start, stop, num)
        '''
        for j in np.arange(min,max+((max-min)/1000),(max-min)/1000):
            ml_data.append(j)
            graph_data.append(((j/max) * 100))
        final[i] = ml_data #1000 datapoints to plot?
        #created 'a'
        dfs = pd.DataFrame({name: [False] * len(ml_data) for name in bool_cols})
        final = pd.concat([final,dfs], axis=1)
        #added h and i
        for j in [c for c in names if c != i]:
            final[j]=num_cols.loc[j,"avg"] #probably better way to do this
        #adds the rest of numbers
        
        fig, ax = plt.subplots()
        final = final.reindex(columns=featcols)

        severity_pred = model.predict(final)
        plt.scatter(graph_data,severity_pred)
        plt.grid(True)
        plt.xlabel(str(i))
        plt.ylabel("Severity Prediction")
        plt.title("How " + str(i) + " influences model predictions")
        plt.savefig("data_organized\\"+str(i)+"_predictions_from_features.jpg", dpi=600)
        plt.close(fig)
        
        
    
    
    


if __name__ == "__main__":
    model = None
    model2 = None
    #generate new model using True, otherwise leave false to read stored model
    newmodel = False
    newmodel2 = False
    feat,label,featcols = prep_data(pd.concat(
        map(pd.read_csv, ['data\output_0.csv', 'data\output_1.csv', 'data\output_2.csv', 'data\output_3.csv', 'data\output_4.csv', 'data\output_5.csv', 'data\output_6.csv', 'data\output_7.csv'])
        ))
    
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
    model2 = model_RFC(feat,label)
    print("GOGOGOGO")
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
    #create_all_features(feat,label,model2,featcols)