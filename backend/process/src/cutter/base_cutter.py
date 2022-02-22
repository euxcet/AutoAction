import abc

# removed
class BaseCutter(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def cut(self, sensor_data, timestamp, cut_range):
        pass