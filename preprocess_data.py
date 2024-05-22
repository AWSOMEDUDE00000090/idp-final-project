import pandas as pd
import geopandas as gpd

def getrawgdf():
    df = pd.concat(
    map(pd.read_csv, ['data\output_0.csv', 'data\output_1.csv', 'data\output_2.csv', 'data\output_3.csv', 'data\output_4.csv', 'data\output_5.csv', 'data\output_6.csv', 'data\output_7.csv'])
    )
    return df

def getcountry():
    country = gpd.read_file("data\gz_2010_us_040_00_5m.json")
    return country

def getrawhighgdf():
    highgdf = pd.concat(
        map(gpd.read_file, ['data\\National_Highway_System_(NHS)_1.csv', 'data\\National_Highway_System_(NHS)_2.csv', 'data\\National_Highway_System_(NHS)_3.csv', 'data\\National_Highway_System_(NHS)_4.csv'])
        )
    return highgdf

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
    features = pd.get_dummies(features) #out of memory error on full dataset
    print("finished prepping ml data")
    return features, labels

def getmldf():#probably could have used for loop but copy paste was faster
    #, 'data_organized\\features_6.csv', 'data_organized\\features_7.csv'
    #, 'data_organized\\labels_6.csv', 'data_organized\\labels_7.csv
    hdf_files = ['data_organized\\features_0.h5', 'data_organized\\features_1.h5', 'data_organized\\features_2.h5', 'data_organized\\features_3.h5', 'data_organized\\features_4.h5', 'data_organized\\features_5.h5']

    # Define the list of labels HDF files
    labels_hdf_files = ['data_organized\\labels_0.h5', 'data_organized\\labels_1.h5', 'data_organized\\labels_2.h5', 'data_organized\\labels_3.h5', 'data_organized\\labels_4.h5', 'data_organized\\labels_5.h5']

    features = pd.concat([pd.read_hdf(file) for file in hdf_files]) #numpy.core._exceptions._ArrayMemoryError: Unable to allocate 15.0 GiB for an array with shape (45071, 44806) and data type object
    labels = pd.concat([pd.read_hdf(file) for file in labels_hdf_files])

    #df = pd.concat([features_df, labels_df], axis=1)

    return features,labels

def generate_organized_df():
    raw = ['data\output_0.csv', 'data\output_1.csv', 'data\output_2.csv', 'data\output_3.csv', 'data\output_4.csv', 'data\output_5.csv']
    #'data\output_6.csv', 'data\output_7.csv' #are broken, some internal hdf5 error that would take a lot of weird parameter tunning to fix, for now i just discarded it
    count = 0
    for i in raw:
        print("parsing df: "+i)
        features,labels = prep_data(pd.read_csv(i))
        print("start writing features to .h5 for df: "+i)
        #features.to_csv('data_organized/features_'+str(count)+'.csv') #csv took too long
        features.to_hdf('data_organized/features_'+str(count)+'.h5', key='ml', mode='w')
        print("start writing labels to .h5 for df: "+i)
        labels.to_hdf('data_organized/labels_'+str(count)+'.h5', key='ml', mode='w')
        #labels.to_csv('data_organized/labels_'+str(count)+'.csv')  #csv took too long
        count+=1

if __name__ == "__main__":
    #generate_organized_df()
    f,l = getmldf()
    print(f.head())
    print(l.head())