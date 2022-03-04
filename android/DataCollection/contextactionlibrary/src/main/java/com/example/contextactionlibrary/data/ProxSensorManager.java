package com.example.contextactionlibrary.data;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

public class ProxSensorManager extends MySensorManager implements SensorEventListener {

    private Context mContext;

    private Preprocess preprocess;

    private SensorManager mSensorManager;
    private Sensor mProx;
    private int samplingPeriod;

    public ProxSensorManager(Context context, int samplingPeriod, String name) {
        this.mContext = context;
        this.samplingPeriod = samplingPeriod;
        this.setName(name);
        preprocess = Preprocess.getInstance();
        initialize();
    }

    public boolean initialize() {
        if (mSensorManager == null) {
            mSensorManager = (SensorManager) mContext.getSystemService(Context.SENSOR_SERVICE);
            mProx = mSensorManager.getDefaultSensor(Sensor.TYPE_PROXIMITY);
        }

        if (!isSensorSupport()) {
            Log.e(TAG, "Proximity sensor is not supported in this phone.");
            // ToastUtils.showInWindow(mContext, "没有检测到相关传感器");
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
        int type = event.sensor.getType();
        if (type == Sensor.TYPE_PROXIMITY)
            preprocess.preprocessProx(event.values[0], event.timestamp);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {}
}
