from abc import ABC, abstractmethod


class CalibrationScorer(ABC):
    @abstractmethod
    def calc_score(self, x, y):
        pass

    @abstractmethod
    def get_prediction_set(self, x, calibration_score):
        pass
