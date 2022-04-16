package com.hcifuture.contextactionlibrary.sensor.data;

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.List;

public class IMUData extends Data {
    private List<SingleIMUData> data;

    public IMUData() {
        data = new ArrayList<>();
    }

    public List<SingleIMUData> getData() {
        return data;
    }

    public SingleIMUData getLastData() {
        if (data.isEmpty()) {
            return null;
        }
        return data.get(data.size() - 1);
    }

    public void setData(List<SingleIMUData> data) {
        this.data = data;
    }

    public synchronized IMUData tail(int length) {
        if (length > data.size()) {
            data = data.subList(data.size() - length, data.size());
        }
        return this;
    }

    public synchronized void insert(SingleIMUData d, int limit) {
        if (limit > 0) {
            while (data.size() >= limit) {
                data.remove(0);
            }
        }
        data.add(d);
    }

    public void clear() {
        data.clear();
    }

    public synchronized IMUData deepClone() {
        IMUData result = new IMUData();
        for (SingleIMUData d: data) {
            result.insert(d.deepClone(), -1);
        }
        return result;
    }

    @Override
    public DataType dataType() {
        return DataType.IMUData;
    }
}
