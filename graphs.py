import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
#import manim doesn't work we are using plotly lmao
import plotly as plo
import plotly.graph_objects as go
import plotly.express as px
#cool idea, https://stackoverflow.com/questions/6687660/keep-persistent-variables-in-memory-between-runs-of-python-script we gonna make wrapper for
#data, and it can rerun new script
import mach_learning as ml

#https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html
#Suggested Citation:				
#Annual Estimates of the Resident Population for the United States, Regions, States, District of Columbia, and Puerto Rico: April 1, 2020 to July 1, 2022 (NST-EST2022-POP)				
#Source: U.S. Census Bureau, Population Division				
#Release Date: December 2022				
def getgdf():
    df = pd.concat(
    map(pd.read_csv, ['data\output_0.csv', 'data\output_1.csv', 'data\output_2.csv', 'data\output_3.csv', 'data\output_4.csv', 'data\output_5.csv', 'data\output_6.csv', 'data\output_7.csv'])
    )
    return df

def getcountry():
    country = gpd.read_file("data\gz_2010_us_040_00_5m.json")
    return country

def gethighgdf():
    highgdf = pd.concat(
        map(gpd.read_file, ['data\\National_Highway_System_(NHS)_1.csv', 'data\\National_Highway_System_(NHS)_2.csv', 'data\\National_Highway_System_(NHS)_3.csv', 'data\\National_Highway_System_(NHS)_4.csv'])
        )
    return highgdf

def makeGraphs(df,country,highgdf):
    #machine learning test
    #f,v = ml.prep_data(df)
    #model = ml.model_regression(f,v,0.3)
    '''
    df2= pd.read_csv("states.csv")

    y = sorted(df["State"].unique())
    x = sorted(df2["Abbreviation"].unique())
    print()
    print("We got a csv with state abbrieviations to compare to our dataset after some confusion from 'https://github.com/jasonong/List-of-US-States/blob/master/states.csv'")
    print("Our database lacks a few sates: database:"+str(len(y))+" < all states+dc:"+str(len(x)))
    print("States we dont have data for: " + str([i for i in x if not i in y]))
    print("___________________")
    print("Both databases have DC, but our dataset claims to have 49 states while lacking Alaska and Hawaii which means it actually has 48, meaning the page was wrong.")
    print("So we decided to delete DC, as its not a state.")
    '''

    country = country[(country['NAME'] != 'District of Columbia') & (country['NAME'] != 'Puerto Rico') & (country['NAME'] != 'Alaska') & (country['NAME'] != 'Hawaii')]

    fig, ax = plt.subplots(1, figsize=(12,5))

    country.boundary.plot(ax=ax,color="black")

    gdf2 = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Start_Lng,df.Start_Lat), crs="EPSG:4326"
    )

    #turn off tick lables, as they aren't usefull
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title('Crash Locations in the US visualized (February 2016 to Dec 2020)')
    gdf2.plot(aspect=1,ax=ax,markersize=0.1,cmap='PuBu') #column="POP2010"
    plt.savefig('america.jpg', dpi=600)

    #show crashes per state
    temp = df.groupby('State')['ID'].count()
    df2= pd.read_csv("states.csv")
    df2 = df2.merge(temp, left_on='Abbreviation', right_on='State',how="left")
    print(df2.head())
    fig, ax = plt.subplots()
    #ax.pie(temp, labels=temp.index,autopct='%1.1f%%')
    country2 = country.merge(df2, left_on='NAME', right_on='State',how="left")
    print(country2.head())
    plt.title('Total Crash per state in the US visualized (February 2016 to Dec 2020)')
    #TODO color by severity
    #turn off tick lables, as they aren't usefull
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    country2.plot(column = 'ID',ax=ax,legend=True)

    #plt.xlabel('x label')
    #plt.ylabel('y label')
    plt.title('Total Crashes Per State (February 2016 to Dec 2020)')

    plt.savefig('states.jpg', dpi=600)
    #graph looking at if states with massive # of crashes corolate to population
    #goal of figuring out what conditions are most common in states with most crashes
    #adjust for population of state (most recent data for pop is 2019)
    df3= pd.read_csv("NST-EST2022-POP.csv")
    country2 = country2.merge(df3, left_on='NAME', right_on='Geographic Area',how="left")
    #February 2016 to Dec 2020 crash data, using 2020 census #
    country2["2020"] = country2["2020"].apply(lambda x: int(x.replace(",", "")))
    #TODO need to limit Start_Time to 2020
    country2["crashsesperpop"] = country2["ID"] / country2["2020"]
    print(country2.head())
    fig, ax = plt.subplots()
    #turn off tick lables, as they aren't usefull
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    country2.plot(column = 'crashsesperpop',ax=ax,legend=True)
    plt.title('Crashes Per Person Per State (2020)')
    plt.savefig('statesadjusted.jpg', dpi=600)

    #Bar Plot of # of crashes based on weather conditions
    #df['ID'].count()
    #Bar plot with different types of roads and the # of accidents
    common = df.loc[:,['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']].value_counts().reset_index()

    def makename(row):
        vals = ['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']
        out = ""
        for i,v in zip(row,vals):
            if i:
                if out != "":
                    out += " and "+ v
                else:
                    out+= v
        if out == "":
            return "Nothing"
        else:
            return out
    
    def addlabels(x,y):
        for i in range(len(x)):
            loc = y[i]
            if i == 0:
                loc = loc //2
            plt.text(i, loc, x[i] + f" ({y[i]})",rotation = 90,ha = 'center',va='bottom')
    common["Name"] = common.apply(makename,axis=1)
    print(common.head())
    common = common.nlargest(10, 'count')
    fig, ax = plt.subplots()
    #plt.xticks(rotation=90)
    ax.set_xticklabels([])
    plt.title('10 most common elements near a crash (February 2016 to Dec 2020)')
    addlabels(common["Name"], common['count'])
    ax.bar(common["Name"], common['count'])#, label=bar_labels, color=bar_colors
    plt.savefig('situations.jpg', dpi=600)
    #plt.bar(courses, values, color ='maroon', width = 0.4)
    
    #Interactive Graphs
    #would be nice to take the ones below 5% and plot them on a different pie, cuz its so small, while unifying them in the bigger pie
    temp = df['Visibility(mi)'].value_counts()
    print(temp.head())
    fig = px.pie(temp, values='count', names=temp.index, title='Percentage of accidents with visibilities')
    fig.show()
    #temp2 = temp[temp.index != 10]

    fig = px.violin(df, x='Severity', y='Visibility(mi)') #, render_mode='webgl'
    fig.update_traces(marker_color='green')
    fig.show()

    #most dangerous times to drive per state
    
    #Our data has 5 fundemental catagories of information about crashes, our whole project
    #is about showing how these 4 different factors contribute to the outcomes of a crash
    #when = ['Start_Time','Sunrise_Sunset']
    #where = ['Start_Lat','Start_Lat','State'] #states graph, and united states map DONE
    #weather = ['Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Weather_Condition'] #visibility interactive graph
    #road_elements = ['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop'] #graph showing 10 most common road elements
    #-------------------------------------------------------#
    #outcomes = ['Distance(mi)','Severity','End_Time','Start_Time'] #End_Time-Start_Time = Duration, Data existing at all means it was crash
    


    plt.close()
    print("End")