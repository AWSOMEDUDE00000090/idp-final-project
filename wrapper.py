import sys
import graphs
from importlib import reload
import time

df = None
country = None
highgdf = None
if __name__ == "__main__":
    while True:
        #we load our gdfs, into ram in wrapper so we can edit
        #main script and not need to waste time loading data
        #these methods should never take parameters (unless they are file names), their job is ONLY to load the data
        #and keep it in this extra python process
        #all the processing and plotting is done in makeGraphs
        print("Starting process!")
        
        if df == None:
            start = time.perf_counter()
            print("Loading df into ram!")
            df = graphs.getgdf()
            end = time.perf_counter()
            print("df loaded! took:",(end-start))
        if country == None:
            start = time.perf_counter()
            print("Loading country into ram!")
            country = graphs.getcountry()
            end = time.perf_counter()
            print("country loaded! took:",(end-start))
        if highgdf == None:
            start = time.perf_counter()
            print("Loading highway data into ram! estimated time: 185 sec or 3 min")
            highgdf = graphs.gethighgdf()
            end = time.perf_counter()
            print("highway data loaded! took:",(end-start))
        print("Starting main!")
        try:
            graphs.makeGraphs(df,country,highgdf)
        except:
            print("graphs has failed")
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        reload(graphs) #needed to kill process and load the new code