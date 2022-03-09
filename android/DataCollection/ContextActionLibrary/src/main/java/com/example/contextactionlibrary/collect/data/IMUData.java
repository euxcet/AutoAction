package com.example.contextactionlibrary.collect.data;

import java.util.ArrayList;
import java.util.List;

public class IMUData extends Data {
    private List<List<Float>> data;

    public IMUData() {
        data= new ArrayList<>();
    }

    public List<List<Float>> getData() {
        return data;
    }

    public List<Float> getLastData() {
        return data.get(data.size() - 1);
    }

    public void setData(List<List<Float>> data) {
        this.data = data;
    }

    public synchronized void insert(List<Float> d, int limit) {
        while (data.size() >= limit) {
            data.remove(0);
        }
        data.add(d);
    }

    public synchronized List<Float> toList() {
        List<Float> list = new ArrayList<>();
        for(List<Float> l: data) {
            list.addAll(l);
        }
        return list;
    }

    public void clear() {
        data.clear();
    }
}
