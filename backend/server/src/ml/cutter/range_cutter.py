class RangeCutter():
    def __init__(self):
        pass

    def cut(self, sensors_data, cut_range):
        result = []
        for data in sensors_data:
            single = []
            for r in cut_range:
                single.append(data[r[0] : r[1]])
            result.append(single)
        return result