class State:
    _instance = None

    def __init__(self):
        if not hasattr(self, 'initialized'):
            # 운전자
            self.attention = 0
            self.attentionMinLevel = 0
            self.attentionMaxLevel = 0
            # 스피커
            self.isMuted = False
            self.soundLevel = 0
            # 온도
            self.carTemp = 0
            self.driverTemp = 0
            self.passengerTemp = 0
            self.carAverageTemp = 0
            self.driverAverageTemp = 0
            self.passengerAverageTemp = 0
            self.initizlized = True

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(State, cls).__new__(cls)
        return cls._instance
    
    