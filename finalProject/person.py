import time
from timecalc import timer


class person_in_pool:
    #initializes a person with ID and a distance, and checks if the person is underwater
    def __init__(self,id,distance,watersurface):
        self.id = id
        self.distance = distance
        self.isUnderWater = (self.distance > watersurface)
        self.timer = timer(self.isUnderWater)

    #updates the person's distance
    def updateDistance(self,newDistance,watersurface):
        before_update = self.distance > watersurface # if he ws underwater before update
        self.distance = newDistance
        self.isUnderWater = (newDistance > watersurface)
        if self.isUnderWater != before_update:
            self.timer.setTime(self.isUnderWater)


    #checks if the person is drowning based on the current distance ansd the duration
    def isDrowning(self,currdist,watersurface):
        self.updateDistance(currdist,watersurface)
        if self.isUnderWater is True:
            return self.timer.is_elapsed()
        return False

