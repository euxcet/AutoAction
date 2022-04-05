package com.hcifuture.contextactionlibrary.collect.listener;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;

import com.hcifuture.contextactionlibrary.collect.collector.SensorCollector;

public class NonIMUSensorEventListener implements SensorEventListener {

    SensorCollector collector;
    int sensorType;

    public NonIMUSensorEventListener(SensorCollector collector, int sensorType) {
        this.collector = collector;
        this.sensorType = sensorType;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        this.collector.addSensorData(event.values[0], 0, 0, sensorType, event.timestamp);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }
}
