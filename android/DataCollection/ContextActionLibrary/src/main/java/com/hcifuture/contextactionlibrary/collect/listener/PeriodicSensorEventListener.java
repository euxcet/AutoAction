package com.hcifuture.contextactionlibrary.collect.listener;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;

import com.hcifuture.contextactionlibrary.collect.collector.SensorCollector;

public class PeriodicSensorEventListener implements SensorEventListener {

    private int sensorType;
    private SensorCollector collector;

    public PeriodicSensorEventListener(SensorCollector collector, int sensorType, int period) {
        this.sensorType = sensorType;
        this.collector = collector;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        this.collector.addSensorData(event.values[0], event.values[1], event.values[2], sensorType, (long)(event.timestamp / 1e6));
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }
}
