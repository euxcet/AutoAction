package com.example.datacollection.contextaction.sensor;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import java.lang.reflect.Method;

public class ProximitySensorManager extends MySensorManager implements SensorEventListener {

    private Context mContext;

    private SensorManager mSensorManager;
    private Sensor mProx;
    private int samplingPeriod;

    public ProximitySensorManager(Context context, int samplingPeriod, String name, Object container, Method onSensorChanged) {
        this.mContext = context;
        this.samplingPeriod = samplingPeriod;
        this.name = name;
        this.container = container;
        this.onSensorChanged = onSensorChanged;
        initialize();
    }

    public boolean initialize() {
        if (mSensorManager == null) {
            mSensorManager = (SensorManager) mContext.getSystemService(Context.SENSOR_SERVICE);
            mProx = mSensorManager.getDefaultSensor(Sensor.TYPE_PROXIMITY);
        }

        if (!isSensorSupport()) {
            Log.e(TAG, "Proximity sensor is not supported in this phone.");
            return false;
        }

        isSensorOpened = true;

        isInitialized = true;

        return true;
    }

    public void registerSensorListener() {
        if (isSensorOpened) {
            mSensorManager.registerListener(this, mProx, samplingPeriod);
        }
    }

    public void unRegisterSensorListener() {
        if (isSensorOpened) {
            mSensorManager.unregisterListener(this, mProx);
        }
    }

    public boolean isSensorSupport() {
        return mProx != null;
    }

    @Override
    public synchronized void start() {
        if (isStarted) {
            Log.d(TAG, name + " is already started.");
            return;
        }
        if (!isInitialized) {
            if (!initialize()) {
                return;
            }
        }
        registerSensorListener();
        isStarted = true;
    }

    @Override
    public synchronized void stop() {
        if (!isStarted) {
            Log.d(TAG, name + " is already stopped");
            return;
        }
        unRegisterSensorListener();
        isStarted = false;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        try {
            onSensorChanged.invoke(container, event);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {}
}
