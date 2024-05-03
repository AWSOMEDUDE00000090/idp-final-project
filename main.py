import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
df = pd.concat(
    map(pd.read_csv, ['output_0.csv', 'output_1.csv', 'output_2.csv', 'output_3.csv', 'output_4.csv', 'output_5.csv', 'output_6.csv', 'output_7.csv'])
)

country = gpd.read_file("C:\\Users\\1109367\\idp-final-project\\gz_2010_us_040_00_5m.json")
country = country[(country['NAME'] != 'District of Columbia') & (country['NAME'] != 'Puerto Rico')]
country.axis("off")
country.plot()
plt.savefig('america.jpg', dpi=600)

def draw_us_inset(gdf, **args):
    # Create a new figure and axes for the main plot
    fig, ax = plt.subplots(1, figsize=(12,5))

    # Plot the entire United States
    gdf.plot(ax=ax, **args)

    # Set the extent of the main plot to cut off Alaska & Hawaii
    # ax.set_xlim(-130, -65)
    # ax.set_ylim(23, 50)

    # helper method to plot both insets without a border, no legend, correct vmin/vmax
    def inset(position, name, xlim=None):
        '''
        position = a list: [x, y, width, height] in % of figure size
        '''
        ax = fig.add_axes(position)
        state = gdf[gdf['NAME'] == name]
        args_no_legend = args
        args_no_legend['legend'] = False
        # The new plot for the inset needs to have the vmin/vmax set to the same
        # values as the main plot so the colors show correctly
        if 'column' in args:
            args_no_legend['vmin'] = gdf[args['column']].min()
            args_no_legend['vmax'] = gdf[args['column']].max()
            
        state.plot(ax=ax, **args_no_legend)
        
        # remove box around inset in 1-line comprehension
        [ ax.spines[side].set_visible(False) for side in ['top', 'bottom', 'left', 'right'] ]
        
        # clip our inset if needed
        if xlim:
            ax.set_xlim(xlim)
        ax.set_xticks([])
        ax.set_yticks([])

    inset([0.20, 0.20, 0.20, 0.20], 'Alaska', xlim=(-180, -130))
    inset([0.37, 0.19, 0.05, 0.15], 'Hawaii')

draw_us_inset(country)
country.savefig("inset.jpg")