package com.hcifuture.contextactionlibrary.sensor.collector.sync;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.provider.Settings;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorListener;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.data.Data;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class NonIMUCollector extends SynchronousCollector implements SensorEventListener {
    // TODO: split NonIMUCollector into collectors for each sensor.

    private NonIMUData data;
    private SensorManager sensorManager;
    private Sensor mPressure;
    private Sensor mLight;
    private Sensor mProximity;
    private Sensor mStepCounter;

    public NonIMUCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        data = new NonIMUData();
    }


    @Override
    public void initialize() {
        sensorManager = (SensorManager) mContext.getSystemService(Context.SENSOR_SERVICE);

        mPressure = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE);
        mLight = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT);
        mProximity = sensorManager.getDefaultSensor(Sensor.TYPE_PROXIMITY);
        mStepCounter = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_COUNTER);

        this.resume();
    }

    @Override
    public void close() {
        sensorManager.unregisterListener(this);
    }

    @Override
    public synchronized void pause() {
        sensorManager.unregisterListener(this);
    }

    @Override
    public synchronized void resume() {
        sensorManager.registerListener(this, mPressure, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this, mLight, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this, mProximity, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this, mStepCounter, SensorManager.SENSOR_DELAY_NORMAL);
    }

    @Override
    public String getName() {
        return "NonIMU";
    }

    @Override
    public String getExt() {
        return ".json";
    }

    @Override
    public synchronized CollectorResult getData(TriggerConfig config) {
        data.setScreenBrightness(Settings.System.getInt(mContext.getContentResolver(),Settings.System.SCREEN_BRIGHTNESS,125));
        data.setScreenBrightnessTimestamp(System.currentTimeMillis());
        CollectorResult result = new CollectorResult();
        result.setDataString(gson.toJson(data));
        result.setData(gson.fromJson(result.getDataString(), NonIMUData.class));
        return result;
    }

    /*
    @Override
    public synchronized String getDataString(TriggerConfig config) {
        data.setScreenBrightness(Settings.System.getInt(mContext.getContentResolver(),Settings.System.SCREEN_BRIGHTNESS,125));
        data.setScreenBrightnessTimestamp(System.currentTimeMillis());
        return gson.toJson(data);
    }
     */

    @Override
    public synchronized void onSensorChanged(SensorEvent event) {
        if (data != null) {
            switch (event.sensor.getType()) {
                case Sensor.TYPE_PRESSURE:
                    data.setAirPressure(event.values[0]);
                    data.setAirPressureTimestamp(event.timestamp);
                    break;
                case Sensor.TYPE_LIGHT:
                    data.setEnvironmentBrightness(event.values[0]);
                    data.setEnvironmentBrightnessTimestamp(event.timestamp);
                    break;
                case Sensor.TYPE_PROXIMITY:
                    data.setProximity(event.values[0]);
                    data.setProximityTimestamp(event.timestamp);
                    break;
                case Sensor.TYPE_STEP_COUNTER:
                    data.setStepCounter(event.values[0]);
                    data.setStepCounterTimestamp(event.timestamp);
                    break;
                default:
                    break;
            }
        }
        for (CollectorListener listener: listenerList) {
            listener.onSensorEvent(data);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
