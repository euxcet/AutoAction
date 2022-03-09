package com.example.contextactionlibrary.data;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import com.example.contextactionlibrary.contextaction.action.ActionBase;

import java.util.List;

public class AlwaysOnSensorManager extends MySensorManager implements SensorEventListener {

    private Context mContext;

    private Preprocess preprocess;

    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor mGyroscope;
    private Sensor mLinear;
    private Sensor mMagnetic;
    private int samplingPeriod;
    private List<ActionBase> actions;

    public AlwaysOnSensorManager(Context context, int samplingPeriod, String name, List<ActionBase> actions) {
        this.mContext = context;
        this.samplingPeriod = samplingPeriod;
        this.setName(name);
        this.actions = actions;
        preprocess = Preprocess.getInstance();
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
            // ToastUtils.showInWindow(mContext, "没有检测到相关传感器");
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
        for (ActionBase action: actions) {
            action.onAlwaysOnSensorChanged(event);
        }
        int type = event.sensor.getType();
        switch (type) {
            case Sensor.TYPE_GYROSCOPE:
            case Sensor.TYPE_ACCELEROMETER:
                preprocess.preprocessIMU(type, event.values[0], event.values[1], event.values[2], event.timestamp);
                break;
            default:
                break;
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {}
}
