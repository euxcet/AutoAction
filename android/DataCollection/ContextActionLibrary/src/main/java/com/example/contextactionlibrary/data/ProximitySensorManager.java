package com.example.contextactionlibrary.data;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import com.example.contextactionlibrary.contextaction.action.ActionBase;
import com.example.contextactionlibrary.contextaction.context.ContextBase;

import java.util.List;

public class ProximitySensorManager extends MySensorManager implements SensorEventListener {

    private Preprocess preprocess;

    private SensorManager mSensorManager;
    private Sensor mProximity;
    private int samplingPeriod;

    public ProximitySensorManager(Context context, String name, List<ActionBase> actions, List<ContextBase> contexts, int samplingPeriod) {
        super(context, name, actions, contexts);
        this.samplingPeriod = samplingPeriod;
        preprocess = Preprocess.getInstance();
        initialize();
    }

    public boolean initialize() {
        if (mSensorManager == null) {
            mSensorManager = (SensorManager) mContext.getSystemService(Context.SENSOR_SERVICE);
            mProximity = mSensorManager.getDefaultSensor(Sensor.TYPE_PROXIMITY);
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
            mSensorManager.registerListener(this, mProximity, samplingPeriod);
        }
    }

    public void unRegisterSensorListener() {
        if (isSensorOpened) {
            mSensorManager.unregisterListener(this, mProximity);
        }
    }

    public boolean isSensorSupport() {
        return mProximity != null;
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
    public void onSensorChangedDex(SensorEvent event) {
        onSensorChanged(event);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        for (ActionBase action: actions) {
            action.onProximitySensorChanged(event);
        }
        for (ContextBase context: contexts) {
            context.onIMUSensorChanged(event);
        }
        int type = event.sensor.getType();
        if (type == Sensor.TYPE_PROXIMITY) {
            preprocess.preprocessProx(event.values[0], event.timestamp);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {}
}