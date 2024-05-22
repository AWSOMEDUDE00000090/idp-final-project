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

    #i did a little clever find and replace, but this sucks to look at
    #,common['Bump'],common['Crossing'],common['Give_Way'],common['Junction'],common['No_Exit'],common['Railway'],common['Roundabout'],common['Station'],common['Stop'],common['Traffic_Calming'],common['Traffic_Signal'],common['Turning_Loop']
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
    

    '''
    cmap vals: 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 
    'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG',
    'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu',
    'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
    'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr',
    'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r',
    'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg',
    'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r',
    'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 
    'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 
    'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
    '''
    #passing missing_kwds one can specify the style and label of features containing None or NaN.
    '''
    missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "Missing values",
        },
    '''
    #temp = df.groupby('Severity')['Visibility(mi)'].unique()

    #would be nice to take the ones below 5% and plot them on a different pie, cuz its so small, while unifying them in the bigger pie
    temp = df['Visibility(mi)'].value_counts()
    print(temp.head())
    fig = px.pie(temp, values='count', names=temp.index, title='Percentage of accidents with visibilities')
    fig.show()
    #temp2 = temp[temp.index != 10]

    fig = px.violin(df, x='Severity', y='Visibility(mi)') #, render_mode='webgl'
    fig.update_traces(marker_color='green')
    fig.show()

    '''
    # Create figure
    fig = go.Figure()

    # Add surface trace
    fig.add_trace(go.Heatmap(z=df.values.tolist(), colorscale="Viridis"))

    # Update plot sizing
    fig.update_layout(
        width=800,
        height=900,
        autosize=False,
        margin=dict(t=100, b=0, l=0, r=0),
    )

    # Update 3D scene options
    fig.update_scenes(
        aspectratio=dict(x=1, y=1, z=0.7),
        aspectmode="manual"
    )

    # Add dropdowns
    button_layer_1_height = 1.08
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["colorscale", "Viridis"],
                        label="Viridis",
                        method="restyle"
                    ),
                    dict(
                        args=["colorscale", "Cividis"],
                        label="Cividis",
                        method="restyle"
                    ),
                    dict(
                        args=["colorscale", "Blues"],
                        label="Blues",
                        method="restyle"
                    ),
                    dict(
                        args=["colorscale", "Greens"],
                        label="Greens",
                        method="restyle"
                    ),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            ),
            dict(
                buttons=list([
                    dict(
                        args=["reversescale", False],
                        label="False",
                        method="restyle"
                    ),
                    dict(
                        args=["reversescale", True],
                        label="True",
                        method="restyle"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.37,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            ),
            dict(
                buttons=list([
                    dict(
                        args=[{"contours.showlines": False, "type": "contour"}],
                        label="Hide lines",
                        method="restyle"
                    ),
                    dict(
                        args=[{"contours.showlines": True, "type": "contour"}],
                        label="Show lines",
                        method="restyle"
                    ),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.58,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            ),
        ]
    )

    fig.update_layout(
        annotations=[
            dict(text="colorscale", x=0, xref="paper", y=1.06, yref="paper",
                                align="left", showarrow=False),
            dict(text="Reverse<br>Colorscale", x=0.25, xref="paper", y=1.07,
                                yref="paper", showarrow=False),
            dict(text="Lines", x=0.54, xref="paper", y=1.06, yref="paper",
                                showarrow=False)
        ])

    fig.show()
    '''
    #highway vs city stuff, where we compare which one is more likely to get in accident
    #https://stackoverflow.com/questions/69082127/plot-heatmap-kdeplot-with-geopandas
    #https://altair-viz.github.io/gallery/radial_chart.html
    #bts.gov
    #print(highgdf.head())
    
    plt.close()
    print("End")