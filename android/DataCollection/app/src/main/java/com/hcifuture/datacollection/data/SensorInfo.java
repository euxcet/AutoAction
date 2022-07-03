package com.hcifuture.datacollection.data;

import java.util.ArrayList;
import java.util.List;

/**
 * A sensor data unit, recording the idx, x, y, z, and timestamp data.
 * Q: Why organized in scalar units ??? Is this inefficient ???
 */
public class SensorInfo {
    private List<Float> data;
    private long time;

    // idx is actually the sensor type!
    public SensorInfo(float idx, float x, float y, float z, long time) {
        data = new ArrayList<>();
        data.add(idx);
        data.add(x);
        data.add(y);
        data.add(z);
        this.time = time;
    }

    // never used
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

    public List<Float> getData() {
        return data;
    }

    public long getTime() {
        return time;
    }
}
