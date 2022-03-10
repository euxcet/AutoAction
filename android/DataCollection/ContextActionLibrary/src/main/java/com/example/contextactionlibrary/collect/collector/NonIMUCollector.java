package com.example.contextactionlibrary.collect.collector;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Process;
import android.provider.Settings;

import androidx.annotation.RequiresApi;

import com.example.contextactionlibrary.collect.data.Data;
import com.example.contextactionlibrary.collect.data.NonIMUData;
import com.example.contextactionlibrary.collect.listener.NonIMUSensorEventListener;
import com.google.gson.Gson;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;

public class NonIMUCollector extends SensorCollector {

    private NonIMUData data;

    public NonIMUCollector(Context context, String triggerFolder) {
        super(context, triggerFolder);
        data = new NonIMUData();
    }

    @Override
    public synchronized void addSensorData(float x, float y, float z, int idx, long time) {
        if (data != null) {
            switch (idx) {
                case Sensor.TYPE_PRESSURE:
                    data.setAirPressure(x);
                    data.setAirPressureTimestamp(time);
                    break;
                case Sensor.TYPE_LIGHT:
                    data.setEnvironmentBrightness(x);
                    data.setEnvironmentBrightnessTimestamp(time);
                    break;
                default:
                    break;
            }
        }
    }

    private NonIMUSensorEventListener pressureListener;
    private NonIMUSensorEventListener lightListener;

    private Sensor mPressure;
    private Sensor mLight;

    @Override
    public void initialize() {
        sensorManager = (SensorManager) mContext.getSystemService(Context.SENSOR_SERVICE);

        mPressure = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE);
        mLight = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT);

        pressureListener = new NonIMUSensorEventListener(this, Sensor.TYPE_PRESSURE);
        lightListener = new NonIMUSensorEventListener(this, Sensor.TYPE_LIGHT);

        sensorThread = new HandlerThread("NonIMU Thread", Process.THREAD_PRIORITY_MORE_FAVORABLE);
        sensorThread.start();
        sensorHandler = new Handler(sensorThread.getLooper());

        this.resume();
    }

    @Override
    public void setSavePath() {
        if (data instanceof java.util.List)
            saver.setSavePath(new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date()) + ".bin");
        else
            saver.setSavePath(new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date()) + ".txt");
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public synchronized CompletableFuture<Data> collect() {
        if (data == null) {
            return CompletableFuture.completedFuture(null);
        }
        data.setScreenBrightness(Settings.System.getInt(mContext.getContentResolver(),Settings.System.SCREEN_BRIGHTNESS,125));
        data.setScreenBrightnessTimestamp(System.currentTimeMillis());
        saver.save(data.deepClone());
        return CompletableFuture.completedFuture(data);
    }

    @Override
    public void close() {
        sensorManager.unregisterListener(pressureListener);
        sensorManager.unregisterListener(lightListener);
        if (sensorThread != null)
            sensorThread.quitSafely();
    }

    @Override
    public synchronized void pause() {
        sensorManager.unregisterListener(pressureListener);
        sensorManager.unregisterListener(lightListener);
    }

    @Override
    public synchronized void resume() {
        sensorManager.registerListener(pressureListener, mPressure, SensorManager.SENSOR_DELAY_NORMAL, sensorHandler);
        sensorManager.registerListener(lightListener, mLight, SensorManager.SENSOR_DELAY_NORMAL, sensorHandler);
    }

    @Override
    public boolean forPrediction() {
        return true;
    }

    @Override
    public synchronized Data getData() {
        Gson gson = new Gson();
        return gson.fromJson(gson.toJson(data), NonIMUData.class);
    }

    @Override
    public String getSaveFolderName() {
        return "NonIMU";
    }
}