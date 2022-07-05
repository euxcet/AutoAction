package com.hcifuture.datacollection.data;

/**
 * A 3D sensor data unit, stores 3 float values at x, y, z axes
 * and a long timestamp at the moment.
 */
public class SensorData3D {
    public float[] v;
    public long t;

    public SensorData3D(float x, float y, float z, long timestamp) {
        v = new float[] {x, y, z};
        t = timestamp;
    }

    public SensorData3D(float[] values, long timestamp) {
        v = new float[] {values[0], values[1], values[2]};
        t = timestamp;
    }
}
