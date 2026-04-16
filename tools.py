import time
import  math

def time_lapsed(var = None):
    if var == None:
        return time.time()
    else:
        lasped = time.time() - var
        days = math.floor(((lasped/60)/60)/24)
        hours = math.floor(((lasped/60)/60))-60*days
        minutes = math.floor((lasped/60))-60*hours
        sec = math.floor(lasped - ((minutes*60)+(hours*60*60)+(days*24*60*60)))
        listOfItems = [sec,minutes,hours,days]
        outputlist =  [0,0,0,0]

        for item in listOfItems:
            if item > 0:
                outputlist[listOfItems.index(item)] = item
        return outputlist
        
        

