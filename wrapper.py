import sys
import graphs
from importlib import reload

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
        if not df:
            print("Loading df into ram!")
            df = graphs.getgdf()
            print("df loaded!")
        if not country:
            print("Loading country into ram!")
            country = graphs.getcountry()
            print("country loaded!")
        if not highgdf:
            print("Loading highway data into ram!")
            highgdf = graphs.gethighgdf()
            print("highway data loaded!")
        print("Starting main!")
        graphs.makeGraphs(df,country,highgdf)
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        reload(graphs) #needed to kill process and load the new code