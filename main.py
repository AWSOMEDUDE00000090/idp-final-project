import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
df = pd.read_csv("")

country = gpd.read_file("C:\\Users\\1109367\\idp-final-project\\gz_2010_us_040_00_5m.json")
country = country[(country['NAME'] != 'District of Columbia') & (country['NAME'] != 'Puerto Rico')]

country.plot()
plt.savefig('america.jpg', dpi=600)