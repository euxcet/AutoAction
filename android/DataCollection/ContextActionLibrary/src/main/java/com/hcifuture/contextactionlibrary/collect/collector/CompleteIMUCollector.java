package com.hcifuture.contextactionlibrary.collect.collector;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.collect.data.Data;
import com.hcifuture.contextactionlibrary.collect.data.IMUData;
import com.hcifuture.contextactionlibrary.collect.listener.PeriodicSensorEventListener;
import com.google.gson.Gson;
import com.hcifuture.contextactionlibrary.collect.trigger.Trigger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class CompleteIMUCollector extends SensorCollector {
    // For complete data, keep 10s, 10 * 100 * 4 = 4k data
    // in case sampling period is higher, maybe max 500Hz for acc and gyro
    private final long DELAY_TIME = 5000;
    private int size = 12000;

    private final int samplingPeriod;
    private final int collectPeriod;

    private IMUData data;

    private String sensorData;
    private String taptapPoint;
    private Gson gson = new Gson();

    public CompleteIMUCollector(Context context, Trigger.CollectorType type, String triggerFolder, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, int samplingPeriod, int collectPeriod) {
        super(context, type, triggerFolder, scheduledExecutorService, futureList);
        this.samplingPeriod = samplingPeriod;
        this.collectPeriod = collectPeriod;
        this.data = new IMUData();
    }

    public synchronized void addSensorData(float x, float y, float z, int idx, long time) {
        if (data != null) {
            data.insert(new ArrayList<>(Arrays.asList(
                    x, y, z,
                    (float) (time / 1000000),
                    (float) idx
            )), size);
        }
    }
    
    private PeriodicSensorEventListener gyroListener;
    private PeriodicSensorEventListener linearAccListener;
    private PeriodicSensorEventListener accListener;
    private PeriodicSensorEventListener magListener;

    private Sensor mGyroSensor;
    private Sensor mLinearAccSensor;
    private Sensor mAccSensor;
    private Sensor mMagSensor;

    @Override
    public void initialize() {
        sensorManager = (SensorManager) mContext.getSystemService(Context.SENSOR_SERVICE);

        mGyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mLinearAccSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mAccSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mMagSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        gyroListener = new PeriodicSensorEventListener(this, Sensor.TYPE_GYROSCOPE, collectPeriod);
        linearAccListener = new PeriodicSensorEventListener(this, Sensor.TYPE_LINEAR_ACCELERATION, collectPeriod);
        accListener = new PeriodicSensorEventListener(this, Sensor.TYPE_ACCELEROMETER, collectPeriod);
        magListener = new PeriodicSensorEventListener(this, Sensor.TYPE_MAGNETIC_FIELD, collectPeriod);

        /*
        sensorThread = new HandlerThread("Complete Thread", Process.THREAD_PRIORITY_MORE_FAVORABLE);
        sensorThread.start();
        sensorHandler = new Handler(sensorThread.getLooper());
         */

        this.resume();
    }

    public String getRecentData() {
        return sensorData;
    }

    public String getTapTapPoint() {
        return taptapPoint;
    }

    @Override
    public void setSavePath(String timestamp) {
        saver.setSavePath(timestamp + "_imu.bin");
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public synchronized CompletableFuture<Void> collect() {
        Log.e("TapTapCollector", "collect");
        taptapPoint = gson.toJson(data.getLastData());
        CompletableFuture<Void> ft = new CompletableFuture<>();
        futureList.add(scheduledExecutorService.schedule(() -> {
            List<Float> cur = data.toList();
            sensorData = gson.toJson(cur);
            Log.e("TapTapCollector", "size " + cur.size());
            saver.save(cur).whenComplete((v, t) -> ft.complete(null));
        }, DELAY_TIME, TimeUnit.MILLISECONDS));
        return ft;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public synchronized CompletableFuture<Void> collectShort(int head, int tail) {
        Log.e("TapTapCollector", "collect");
        taptapPoint = gson.toJson(data.getLastData());
        CompletableFuture<Void> ft = new CompletableFuture<>();
        futureList.add(scheduledExecutorService.schedule(() -> {
            int length = (head + tail) * size * 5 / 10000;
            List<Float> tmp = data.toList();
            List<Float> cur = tmp.subList(tmp.size() - length, tmp.size());
            sensorData = gson.toJson(cur);
            Log.e("TapTapCollector short", "size " + cur.size());
            saver.save(cur).whenComplete((v, t) -> ft.complete(null));
        }, tail, TimeUnit.MILLISECONDS));
        return ft;
    }

    @Override
    public void close() {
        sensorManager.unregisterListener(gyroListener);
        sensorManager.unregisterListener(linearAccListener);
        sensorManager.unregisterListener(accListener);
        sensorManager.unregisterListener(magListener);
        /*
        if (sensorThread != null)
            sensorThread.quitSafely();
         */
    }

    @Override
    public synchronized void pause() {
        sensorManager.unregisterListener(gyroListener);
        sensorManager.unregisterListener(linearAccListener);
        sensorManager.unregisterListener(accListener);
        sensorManager.unregisterListener(magListener);
    }

    @Override
    public synchronized void resume() {
        sensorManager.registerListener(gyroListener, mGyroSensor, samplingPeriod, sensorHandler);
        sensorManager.registerListener(linearAccListener, mLinearAccSensor, samplingPeriod, sensorHandler);
        sensorManager.registerListener(accListener, mAccSensor, samplingPeriod, sensorHandler);
        sensorManager.registerListener(magListener, mMagSensor, samplingPeriod, sensorHandler);
    }

    @Override
    public boolean forPrediction() {
        return false;
    }

    @Override
    public synchronized Data getData() {
        Gson gson = new Gson();
        return gson.fromJson(gson.toJson(data), IMUData.class);
    }

    @Override
    public String getSaveFolderName() {
        return "CompleteIMU";
    }
}
