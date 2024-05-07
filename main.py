import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
df = pd.concat(
    map(pd.read_csv, ['data\output_0.csv', 'data\output_1.csv', 'data\output_2.csv', 'data\output_3.csv', 'data\output_4.csv', 'data\output_5.csv', 'data\output_6.csv', 'data\output_7.csv'])
)

country = gpd.read_file("data\gz_2010_us_040_00_5m.json")
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
#country.axis("off")
fig, ax = plt.subplots(1, figsize=(12,5))
#ax.set_xlim(-180, -65)
#ax.set_ylim(23, 50)
country.boundary.plot(ax=ax)

gdf2 = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Start_Lng,df.Start_Lat), crs="EPSG:4326"
)
gdf2.plot(aspect=1,ax=ax,markersize=2,cmap='winter') #column="POP2010"
#cmap='cubehelix'
#cmap='Greens'
#cmap='winter'
#OrRd
#passing missing_kwds one can specify the style and label of features containing None or NaN.
'''
missing_kwds={
        "color": "lightgrey",
        "edgecolor": "red",
        "hatch": "///",
        "label": "Missing values",
    },
'''

#highway vs city stuff, where we compare which one is more likely to get in accident
plt.savefig('america.jpg', dpi=600)