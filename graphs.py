import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
from adjustText import adjust_text
#import manim doesn't work we are using plotly lmao
import plotly as plo
import plotly.graph_objects as go
import plotly.express as px
#cool idea, https://stackoverflow.com/questions/6687660/keep-persistent-variables-in-memory-between-runs-of-python-script we gonna make wrapper for
#data, and it can rerun new script
import mach_learning as ml
import os

#https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html
#Suggested Citation:				
#Annual Estimates of the Resident Population for the United States, Regions, States, District of Columbia, and Puerto Rico: April 1, 2020 to July 1, 2022 (NST-EST2022-POP)				
#Source: U.S. Census Bureau, Population Division				
#Release Date: December 2022				
def getgdf():
    # List of file names directly within the 'data' directory
    file_names = [f'data/output_{i}.csv' for i in range(8)]
    
    # Create relative file paths
    file_paths = [os.path.join(file_name) for file_name in file_names]
    
    # Read and concatenate the CSV files
    df = pd.concat(map(pd.read_csv, file_paths))
    
    return df

def getcountry():
    file_path = os.path.join('data', 'gz_2010_us_040_00_5m.json')
    country = gpd.read_file(file_path)
    return country

# def gethighgdf():
#     highgdf = pd.concat(
#         map(gpd.read_file, ['data\\National_Highway_System_(NHS)_1.csv', 'data\\National_Highway_System_(NHS)_2.csv', 'data\\National_Highway_System_(NHS)_3.csv', 'data\\National_Highway_System_(NHS)_4.csv'])
#         )
#     return highgdf

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
    plt.title('Crash Locations in the US visualized February 2016 to Dec 2020')
    gdf2.plot(aspect=1,ax=ax,markersize=0.1,cmap='RdYlGn') #column="POP2010"
    plt.savefig('america.jpg', dpi=600)

    #show crashes per state
    temp = df.groupby('State')['ID'].count()
    df2= pd.read_csv("states.csv")
    df2 = df2.merge(temp, left_on='Abbreviation', right_on='State',how="left")
    fig, ax = plt.subplots()
    #ax.pie(temp, labels=temp.index,autopct='%1.1f%%')
    country2 = country.merge(df2, left_on='NAME', right_on='State',how="left")
    plt.title('Total Crash per state in the US visualized February 2016 to Dec 2020')
    #TODO color by severity
    #turn off tick lables, as they aren't usefull
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    country2.plot(column = 'ID',ax=ax,legend=True,cmap="RdYlGn")

    #plt.xlabel('x label')
    #plt.ylabel('y label')
    plt.title('Total Crashes Per State February 2016 to Dec 2020')

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
    fig, ax = plt.subplots()
    #turn off tick lables, as they aren't usefull
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    country2.plot(column = 'crashsesperpop',ax=ax,legend=True,cmap="RdYlGn")
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
    common2 = common.nlargest(10, 'count')
    fig, ax = plt.subplots()
    #plt.xticks(rotation=90)
    ax.set_xticklabels([])
    plt.title('10 most common elements near a crash February 2016 to Dec 2020')
    addlabels(common2["Name"], common2['count'])
    plt.ylabel('Crash Count')
    ax.bar(common2["Name"], common2['count'])#, label=bar_labels, color=bar_colors
    plt.savefig('situations.jpg', dpi=600)

    #TODO FIX FIX FIX FIX FIX 

    common3=common.drop(common[common['Name'] == 'Nothing'].index)
    fig, ax = plt.subplots()
    #plt.xticks(rotation=90)
    ax.set_xticklabels([])
    plt.title('10 most common elements near a crash February 2016 to Dec 2020')
    addlabels(common3["Name"], common3['count'])
    plt.ylabel('Crash Count')
    ax.bar(common3["Name"], common3['count'])#, label=bar_labels, color=bar_colors
    plt.savefig('situations2.jpg', dpi=600)
    #plt.bar(courses, values, color ='maroon', width = 0.4)
    
    #Interactive Graphs
    #TODO probably better to make this a regular graph 
    #would be nice to take the ones below 5% and plot them on a different pie, cuz its so small, while unifying them in the bigger pie
    temp = df['Visibility(mi)'].value_counts()
    
    
    temp = temp.reset_index()
    temp["adjustedval"] = temp["Visibility(mi)"]
    temp["adjustedval"]=temp["adjustedval"].apply(lambda x : round(min(x,10), 0))
    print(temp.head())
    temp = temp.rename(columns={"count": "specific_count"})
    print(temp.head())
    temp2 = temp
    temp = temp.groupby('adjustedval')['specific_count'].sum().reset_index()
    #TODO merge the unique 'visibility' vals back into temp
    print(temp.head())
    #fig = px.pie(temp, values='specific_count', names="adjustedval", title='Percentage of Accidents per Visibility')
    fig = px.sunburst(
    temp,
    names='adjustedval',
    parents='parent',
    values='specific_count',
    )
    
    #TODO uncomment 
    fig.show()
    #temp2 = temp[temp.index != 10]

    #fig = px.violin(df, x='Severity', y='Visibility(mi)') #, render_mode='webgl'
    #fig.update_traces(marker_color='green')
    #TODO uncomment fig.show()

    fig = go.Figure()
    names = ['Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)',
        'Visibility(mi)','Wind_Speed(mph)','Precipitation(in)'] #,'Sunrise_Sunset','Start_Lat','Start_Lng','Start_Time'
    dropdown_buttons = []
    for i,n in enumerate(names):
        temp = df.groupby([n,'Severity']).size().reset_index(name="count")
        fig.add_trace(go.Scatter(mode='markers',x=temp[n], y=temp["count"], marker_color=temp["Severity"],name=n,hovertemplate='<b>%{text}</b><br>' +'Value: %{x}<br>' +'Count: %{y}<br>' +'Severity: %{marker.color}',text=[n]*len(temp),))
        visible = ["legendonly"] * len(names)
        visible[i] = True
        dropdown_buttons.append({'label':n,'method':'update','args':[{'visible' : visible, 'title' : n, 'showlegend' : True}]})
    
    fig.update_layout({'updatemenus':[{'type' : 'dropdown', 'buttons' : dropdown_buttons}]})#'width':800, 'height' : 400, 
    #fig = px.scatter(temp, x="Humidity(%)", y="count", color="Severity", hover_data=['Humidity(%)',"count","Severity"]) #size='petal_length'
    fig.show()
    #TODO remove this
    plt.close('all')
    return
    #TODO I have only made graphs exploring these elements with the idea of the existance of their data showing crashing, i haven't looked into their effect on severity, duration, or distance
    
    #Our data has 5 fundemental catagories of information about crashes, our whole project
    #is about showing how these 4 different factors contribute to the outcomes of a crash
    #when = ['Start_Time','Sunrise_Sunset'] #overall hour plot, per state hour plot
    #where = ['Start_Lat','Start_Lat','State'] #states graph, and united states map DONE
    #weather = ['Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Weather_Condition'] #visibility interactive graph
    #road_elements = ['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop'] #graph showing 10 most common road elements
    #-------------------------------------------------------#
    #outcomes = ['Distance(mi)','Severity','End_Time','Start_Time'] #End_Time-Start_Time = Duration, Data existing at all means it was crash
    

    #just a graph of accidents over time

    #---dataorg
    #"2019-06-12 10:10:56"[0:19] --> "2019-06-12 10:10:56"
    #"2022-12-03 23:37:14.000000000" --> "2022-12-03 23:37:14"
    df["Start_Time"] = df["Start_Time"].apply(lambda x : str(str(x)[0:19]))
    df["Start_Time"] = pd.to_datetime(df['Start_Time']) #same as csv parse dates
    #df['Start_Time'].dt.hour is 0-23, we want to convert to am pm, but its easier to do it in graphing
    hourly_counts = df.groupby(df['Start_Time'].dt.hour).size() #or .count but pretty sure its the same
    #end

    fig, ax = plt.subplots()
    hourly_counts.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('Hour of the day')
    plt.ylabel('Count')
    plt.title('Crash Occurrence Distribution by Hour (Local Time) from February 2016 to Dec 2020') #really annoying name to come up with
    plt.xticks(range(24), [f"{12 if h == 0 else h if h <= 12 else h - 12}:00 {'AM' if h < 12 else 'PM'}" for h in range(24)], rotation=45,ha='right')
    plt.grid(True)
    plt.savefig('whenaccidents.jpg', dpi=600)
    
    #monthly version
    df["Start_Time"] = df["Start_Time"].apply(lambda x : str(str(x)[0:19]))
    df["Start_Time"] = pd.to_datetime(df['Start_Time']) #same as csv parse dates
    #df['Start_Time'].dt.hour is 0-23, we want to convert to am pm, but its easier to do it in graphing
    monthly_counts = df.groupby(df['Start_Time'].dt.month).size() #or .count but pretty sure its the same
    #end

    fig, ax = plt.subplots()
    monthly_counts.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.title('Crash Occurrence Distribution by Month from February 2016 to Dec 2020') #really annoying name to come up with
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    plt.xticks(range(12), months, rotation=45,ha='right')
    plt.grid(True)
    plt.savefig('whenaccidents_month.jpg', dpi=600)

    #when accidents most common per state, hourwise

    #---dataorg
    temp = df.loc[:,['State','Start_Time']]
    temp["Hour"] = df['Start_Time'].dt.hour
    temp = temp.groupby('State')['Hour'].value_counts().reset_index()
    
    max_indices = temp.groupby('State')['count'].idxmax()
    temp = temp.loc[max_indices, :]
    
    
    temp.rename(columns={'State': 'Abbriv'}, inplace=True)
    #TODO fix reading here instead of elsewhere, and the other time i did this
    df2= pd.read_csv("states.csv")
    df2 = df2.merge(temp, left_on='Abbreviation', right_on='Abbriv',how="left")
    country2 = country.merge(df2, left_on='NAME', right_on='State',how="left")
    country2["Hour"] = country2["Hour"].apply(lambda h : f"{12 if int(h) == 0 else int(h) if int(h) <= 12 else int(h) - 12}:00 {'AM' if int(h) < 12 else 'PM'}")
    #end

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.title('Time of Day Most Accidents Happen per State (Local Time)')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    country2.plot(categorical = True,column = 'Hour',ax=ax,legend=True,legend_kwds={'bbox_to_anchor': (1, 0.4)}, cmap='viridis')
    plt.savefig('whenaccidents_state.jpg', dpi=600)


    #when accidents most common per state, monthly

    #---dataorg
    temp = df.loc[:,['State','Start_Time']]
    temp["Month"] = df['Start_Time'].dt.month
    temp = temp.groupby('State')['Month'].value_counts().reset_index()
    
    max_indices = temp.groupby('State')['count'].idxmax()
    temp = temp.loc[max_indices, :]
    
    
    temp.rename(columns={'State': 'Abbriv'}, inplace=True)
    #TODO fix reading here instead of elsewhere, and the other time i did this
    df2= pd.read_csv("states.csv")
    df2 = df2.merge(temp, left_on='Abbreviation', right_on='Abbriv',how="left")
    country2 = country.merge(df2, left_on='NAME', right_on='State',how="left")
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] #also defined
    country2["Month"] = country2["Month"].apply(lambda h : f"{months[int(h)-1]}")
    #end

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.title('Month with Most Accidents Happen per State')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    country2.plot(categorical = True,column = 'Month',ax=ax,legend=True,legend_kwds={'bbox_to_anchor': (1, 0.4)}, cmap='viridis')
    plt.savefig('whenaccidents_state_month.jpg', dpi=600)


    #weather = ['Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Weather_Condition']
    #['Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)']
    #TODO didn't make graph for weather condition yet.
    matplotlib.use('Agg') #required to not run out of memory
    def scattergraph(df,x,xlabel="",ylabel="# of crashes",title="",filename="idk.jpg",outliers_val=-1,grid=True):
        if xlabel == "":
            xlabel = x
        temp = df[x].value_counts().reset_index()
        fig, ax = plt.subplots()
        plt.grid(grid)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.scatter(temp[x],temp["count"])
        if outliers_val != -1:
            outliers = [(i, v) for i, v in zip(temp[x], temp["count"]) if v >= outliers_val]
            outliertext = []
            for x, y in outliers:
                outliertext.append(plt.annotate(f'({x}, {y})', xy=(x, y)))
            adjust_text(outliertext, arrowprops=dict(arrowstyle='->', color='red'))
        plt.savefig(filename, dpi=600)
        plt.close(fig)

    def severitygraph(df,x,xlabel="",ylabel="# of crashes",title="",filename="idk.jpg",outliers_val=-1,grid=True):
        if xlabel == "":
            xlabel = x
        temp = df[x].value_counts().reset_index()
        severity = df[["Severity", x]]
        for i in ["blue", "red", "green", "yellow"]:
            fig, ax = plt.subplots()
            plt.plot(temp[x], temp["count"], color = i)
            plt.grid(grid)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.scatter(temp[x],temp["count"])
            if outliers_val != -1:
                outliers = [(i, v) for i, v in zip(temp[x], temp["count"]) if v >= outliers_val]
                outliertext = []
                for x, y in outliers:
                    outliertext.append(plt.annotate(f'({x}, {y})', xy=(x, y)))
                adjust_text(outliertext, arrowprops=dict(arrowstyle='->', color='red'))
            filename = [i, filename]
            plt.savefig("_".join(filename), dpi=600)
            plt.close(fig)
       
    #can be used to look for connections
    scattergraph(df,'Temperature(F)',title='# of crashes against Temperature',filename='temperture.jpg',outliers_val=11000)
    scattergraph(df,'Wind_Chill(F)',title='# of crashes against Wind Chill',filename='windchill.jpg',outliers_val=7900)
    scattergraph(df,'Humidity(%)',title='# of crashes against Humidity',filename='humidity.jpg',outliers_val=10000)
    scattergraph(df,'Pressure(in)',title='# of crashes against Pressure',filename='pressure.jpg',outliers_val=7500)
    scattergraph(df,'Visibility(mi)',title='# of crashes against Visibility',filename='visibility.jpg',outliers_val=25000)
    scattergraph(df,'Wind_Speed(mph)',title='# of crashes against Wind Speed',filename='windspeed.jpg',outliers_val=15000)
    scattergraph(df,'Precipitation(in)',title='# of crashes against Precipitation',filename='precipitation.jpg',outliers_val=50000)

    severitygraph(df,'Temperature(F)',title='# of crashes against Temperature',filename='temperture.jpg',outliers_val=11000)
    severity = df[["Severity", "Temperature(F)"]]
    for i in range(1,5):
        temp = severity[severity["Severity"== i]]
        plt.plot(temp["x"], temp["Severity"])
        plt.savefig("test.jpg")


    def road_types_bar(df):
        df = df.loc[:, "Bump":"Tuning_Loop"]
        count = (df == True).sum()
        plt.figure(figsize=(10, 6))
        count.plot(kind='bar')
        plt.xlabel('Types of Road')
        plt.ylabel('Number of Crashes')
        plt.title('Road Types Compared With Crashes')
        plt.xticks(rotation=45)
        plt.savefig("roadtypes.jpg")
        
    ## road_types_bar(df)
        
    plt.close('all')
    print("End")