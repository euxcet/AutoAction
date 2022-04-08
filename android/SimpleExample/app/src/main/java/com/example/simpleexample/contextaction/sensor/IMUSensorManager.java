package com.example.simpleexample.contextaction.sensor;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import java.lang.reflect.Method;

public class IMUSensorManager extends MySensorManager implements SensorEventListener {

    private Context mContext;

    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor mGyroscope;
    private Sensor mLinear;
    private Sensor mMagnetic;
    private int samplingPeriod;

    public IMUSensorManager(Context context, int samplingPeriod, String name, Object container, Method onSensorChanged) {
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
            mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
            mLinear = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
            mMagnetic = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        }

        if (!isSensorSupport()) {
            Log.e(TAG, "At least one sensor is not supported in this phone.");
            return false;
        }

        isSensorOpened = true;

        isInitialized = true;

        return true;
    }

    public void registerSensorListener() {
        if (isSensorOpened) {
            mSensorManager.registerListener(this, mAccelerometer, samplingPeriod);
            mSensorManager.registerListener(this, mGyroscope, samplingPeriod);
            mSensorManager.registerListener(this, mLinear, samplingPeriod);
            mSensorManager.registerListener(this, mMagnetic, samplingPeriod);
        }
    }

    public void unRegisterSensorListener() {
        if (isSensorOpened) {
            mSensorManager.unregisterListener(this, mAccelerometer);
            mSensorManager.unregisterListener(this, mGyroscope);
            mSensorManager.unregisterListener(this, mLinear);
            mSensorManager.unregisterListener(this, mMagnetic);
        }
    }

    public boolean isSensorSupport() {
        return mAccelerometer != null && mGyroscope != null && mLinear != null;
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
    public void onAccuracyChanged(Sensor sensor, int i) {
        

    }
}
