package com.hcifuture.datacollection.data;

/**
 * A 1D sensor data unit, stores a float value
 * and a long timestamp at the moment.
 */
public class SensorData1D {
    public float v;
    public long t;

    public SensorData1D(float value, long timestamp) {
        v = value;
        t = timestamp;
    }
}
