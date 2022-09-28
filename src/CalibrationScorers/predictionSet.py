from abc import ABC, abstractmethod

class PredictionSet(ABC):
    @abstractmethod
    def cover(self, y):
        pass
    
    @abstractmethod
    def get_size(self):
        pass 

