import sys
import graphs
import mach_learning as ml
from importlib import reload
import time
import pandas as pd
import traceback

df = pd.DataFrame()
country = pd.DataFrame()
highgdf = pd.DataFrame()
#We dont do anything with highway right now so i disabled it, just change to true to load highway
usehighway = False
if __name__ == "__main__":
    while True:
        #we load our gdfs, into ram in wrapper so we can edit
        #main script and not need to waste time loading data
        #these methods should never take parameters (unless they are file names), their job is ONLY to load the data
        #and keep it in this extra python process
        #all the processing and plotting is done in makeGraphs
        print("Starting process!")
        
        if df.empty:
            start = time.perf_counter()
            print("Loading df into ram!")
            df = graphs.getgdf()
            end = time.perf_counter()
            print("df loaded! took:",(end-start))
        if country.empty:
            start = time.perf_counter()
            print("Loading country into ram!")
            country = graphs.getcountry()
            end = time.perf_counter()
            print("country loaded! took:",(end-start))
        if usehighway and highgdf.empty:
            start = time.perf_counter()
            print("Loading highway data into ram! estimated time: 185 sec or 3 min")
            highgdf = graphs.gethighgdf()
            end = time.perf_counter()
            print("highway data loaded! took:",(end-start))
        print("Starting main!")
        try:
            graphs.makeGraphs(df,country,highgdf)
        except:
            # import module 
            print("graphs has failed")
            traceback.print_exc()
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        reload(ml) #needed for graphs import
        reload(graphs) #needed to kill process and load the new code