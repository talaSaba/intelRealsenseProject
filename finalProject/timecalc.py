import time

class timer:
    #initializes the timer with the current time
    def __init__(self,isunderWater,max_duration = 5):
        self.startTime = time.localtime() if isunderWater else None
        self.max_duration=max_duration


    #calculates the elapsed time and returns True if it exceeds the maximum duration
    def is_elapsed(self):
        if self.startTime==None:
            return False
        currentTime = time.localtime()
        currentTime = currentTime.tm_sec - self.startTime.tm_sec
        if currentTime > self.max_duration:
            return True
        return False

    def printDuration(self):  
        return time.localtime().tm_sec - self.startTime.tm_sec
    def setTime(self,isunderwater):# this function starts the counting if the person is underwater and stops counting if a person is above :)
        if isunderwater == True:# if he now is underwater and was above
            self.startTime = time.localtime()
        else:
            self.startTime = 0

