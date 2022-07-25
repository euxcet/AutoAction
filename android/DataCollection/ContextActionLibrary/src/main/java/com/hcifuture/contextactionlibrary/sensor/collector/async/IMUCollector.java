package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.Session2CommandGroup;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorListener;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorStatusHolder;
import com.hcifuture.contextactionlibrary.sensor.data.IMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.status.Heart;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class IMUCollector extends AsynchronousCollector implements SensorEventListener {
    // For complete data, keep 10s, 10 * 100 * 4 = 4k data
    // in case sampling period is higher, maybe max 500Hz for acc and gyro
    private final long DELAY_TIME = 5000;
    private int LENGTH_LIMIT = 12000;

    private int samplingFrequency = SensorManager.SENSOR_DELAY_FASTEST;

    private IMUData data;

    private SensorManager sensorManager;

    private Sensor mGyroSensor;
    private Sensor mLinearAccSensor;
    private Sensor mAccSensor;
    private Sensor mMagSensor;

    public IMUCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        this.data = new IMUData();
    }

    public void setSamplingFrequency(int freq) {
        samplingFrequency = freq;
    }

    public int getSamplingFrequency() {
        return samplingFrequency;
    }

    @Override
    public void initialize() {
        sensorManager = (SensorManager) mContext.getSystemService(Context.SENSOR_SERVICE);

        mGyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mLinearAccSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mAccSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mMagSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        CollectorStatusHolder.getInstance().setStatus(Sensor.TYPE_GYROSCOPE, mGyroSensor != null);
        CollectorStatusHolder.getInstance().setStatus(Sensor.TYPE_LINEAR_ACCELERATION, mLinearAccSensor!= null);
        CollectorStatusHolder.getInstance().setStatus(Sensor.TYPE_ACCELEROMETER, mAccSensor != null);
        CollectorStatusHolder.getInstance().setStatus(Sensor.TYPE_MAGNETIC_FIELD, mMagSensor != null);

        this.resume();
    }

    @Override
    public void close() {
        if (isRegistered.get()) {
            sensorManager.unregisterListener(this);
            isRegistered.set(false);
        }
    }

    @Override
    public synchronized void pause() {
        if (isRegistered.get()) {
            sensorManager.unregisterListener(this);
            isRegistered.set(false);
        }
    }

    @Override
    public synchronized void resume() {
        if (!isRegistered.get()) {
            sensorManager.registerListener(this, mGyroSensor, samplingFrequency, handler);
            sensorManager.registerListener(this, mLinearAccSensor, samplingFrequency, handler);
            sensorManager.registerListener(this, mAccSensor, samplingFrequency, handler);
            sensorManager.registerListener(this, mMagSensor, samplingFrequency, handler);
            isRegistered.set(true);
        }
    }

    @Override
    public String getName() {
        return "IMU";
    }

    @Override
    public String getExt() {
        return ".bin";
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        Heart.getInstance().newSensorGetEvent(getName(), event.timestamp);
        if (data != null) {
            SingleIMUData newData = new SingleIMUData(
                    Arrays.asList(event.values[0], event.values[1], event.values[2]),
                    event.sensor.getName(),
                    event.sensor.getType(),
                    event.timestamp
            );
            data.insert(newData, LENGTH_LIMIT);
            for (CollectorListener listener: listenerList) {
                listener.onSensorEvent(newData);
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<CollectorResult> getData(TriggerConfig config) {
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        long delay = DELAY_TIME;
        if (config.getImuHead() > 0 && config.getImuTail() > 0) {
            delay = config.getImuTail();
        }
        futureList.add(scheduledExecutorService.schedule(() -> {
            try {
                if (config.getImuHead() > 0 && config.getImuTail() > 0) {
                    int length = (config.getImuHead() + config.getImuTail()) * LENGTH_LIMIT / 10000;
                    CollectorResult result = new CollectorResult();
                    result.setData(data.deepClone().tail(length));
                    ft.complete(result);
                } else {
                    CollectorResult result = new CollectorResult();
                    result.setData(data.deepClone());
                    ft.complete(result);
                }
            } catch (Exception e) {
                e.printStackTrace();
                ft.completeExceptionally(e);
            }
        }, delay, TimeUnit.MILLISECONDS));
        return ft;
    }

    /*
    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<String> getDataString(TriggerConfig config) {
        return getData(config).thenApply((d) -> d == null ? null : gson.toJson(d, IMUData.class));
    }
     */
}
