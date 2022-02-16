package com.example.datacollection.data;

import java.util.ArrayList;
import java.util.List;

public class SensorInfo {
    private List<Float> data;
    private long time;

    public SensorInfo(float idx, float x, float y, float z, long time) {
        data = new ArrayList<>();
        data.add(idx);
        data.add(x);
        data.add(y);
        data.add(z);
        this.time = time;
    }

    public SensorInfo(float idx, float x, float y, float z, float cos, float acc, long time) {
        data = new ArrayList<>();
        data.add(idx);
        data.add(x);
        data.add(y);
        data.add(z);
        data.add(cos);
        data.add(acc);
        this.time = time;
    }

    public long getTime() {
        return time;
    }
}
