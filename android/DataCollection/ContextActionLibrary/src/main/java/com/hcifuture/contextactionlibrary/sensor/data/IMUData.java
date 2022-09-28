package com.hcifuture.contextactionlibrary.sensor.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class IMUData extends Data {
    private final List<SingleIMUData> data;

    public IMUData() {
        data = Collections.synchronizedList(new ArrayList<>());
    }

    public List<SingleIMUData> getData() {
        return data;
    }

    public SingleIMUData getLastData() {
        if (data.isEmpty()) {
            return null;
        }
        synchronized (data) {
            return data.get(data.size() - 1);
        }
    }

    public void removeFirst(int length) {
        synchronized (data) {
            data.subList(0, length).clear();
        }
    }

    public IMUData tail(int length) {
        synchronized (data) {
            if (length < data.size()) {
                removeFirst(data.size() - length);
            }
            return this;
        }
    }

    public void insert(SingleIMUData d, int limit) {
        synchronized (data) {
            if (limit > 0) {
                if (data.size() >= limit) {
                    removeFirst(data.size() - limit + 1);
                }
            }
            data.add(d);
        }
    }

    public void clear() {
        synchronized (data) {
            data.clear();
        }
    }

    public IMUData deepClone() {
        IMUData result = new IMUData();
        synchronized (data) {
            for (SingleIMUData d : data) {
                result.insert(d.deepClone(), -1);
            }
        }
        return result;
    }

    @Override
    public DataType dataType() {
        return DataType.IMUData;
    }
}
