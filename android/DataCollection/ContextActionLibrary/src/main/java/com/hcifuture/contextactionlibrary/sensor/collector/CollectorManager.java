package com.hcifuture.contextactionlibrary.sensor.collector;

import android.content.Context;
import android.util.Log;

import com.hcifuture.contextactionlibrary.sensor.collector.async.AudioCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.BluetoothCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.IMUCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.LocationCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.NonIMUCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.WifiCollector;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.atomic.AtomicBoolean;

public class CollectorManager {
    private Context mContext;
    private ScheduledExecutorService scheduledExecutorService;
    private List<ScheduledFuture<?>> futureList;

    private AtomicBoolean running = new AtomicBoolean(false);
    private List<Collector> collectors = new ArrayList<>();

    public enum CollectorType {
        Audio,
        Bluetooth,
        IMU,
        NonIMU,
        Location,
        Weather,
        Wifi,
        Log,
        All
    }

    private void initializeAll() {
        collectors.add(new BluetoothCollector(mContext, CollectorType.Bluetooth, scheduledExecutorService, futureList));
        collectors.add(new IMUCollector(mContext, CollectorType.IMU, scheduledExecutorService, futureList,  1));
        collectors.add(new NonIMUCollector(mContext, CollectorType.NonIMU, scheduledExecutorService, futureList));
        collectors.add(new WifiCollector(mContext, CollectorType.Wifi, scheduledExecutorService, futureList));
        collectors.add(new LocationCollector(mContext, CollectorType.Location, scheduledExecutorService, futureList));
        collectors.add(new AudioCollector(mContext, CollectorType.Audio, scheduledExecutorService, futureList));
    }

    private void initialize(CollectorType type) {
        switch (type) {
            case Bluetooth:
                collectors.add(new BluetoothCollector(mContext, CollectorType.Bluetooth, scheduledExecutorService, futureList));
                break;
            case IMU:
                collectors.add(new IMUCollector(mContext, CollectorType.IMU, scheduledExecutorService, futureList,1));
                break;
            case NonIMU:
                collectors.add(new NonIMUCollector(mContext, CollectorType.NonIMU, scheduledExecutorService, futureList));
                break;
            case Wifi:
                collectors.add(new WifiCollector(mContext, CollectorType.Wifi, scheduledExecutorService, futureList));
                break;
            case Location:
                collectors.add(new LocationCollector(mContext, CollectorType.Location, scheduledExecutorService, futureList));
                break;
            case Audio:
                collectors.add(new AudioCollector(mContext, CollectorType.Audio, scheduledExecutorService, futureList));
                break;
            case Log:
                Log.e("Trigger", "Do not pass CollectorType.Log in the constructor, it will be ignored.");
                break;
            case All:
                initializeAll();
                break;
            default:
                break;
        }
    }

    public CollectorManager(Context context, List<CollectorType> types, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        this.mContext = context;
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
        try {
            for (CollectorType type : types) {
                initialize(type);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        running.set(true);
    }

    public CollectorManager(Context context, CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        this.mContext = context;
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
        try {
            initialize(type);
        } catch (Exception e) {
            e.printStackTrace();
        }
        running.set(true);
    }

    public void close() {
        for (Collector collector: collectors) {
            collector.close();
        }
    }

    public LogCollector newLogCollector(String label, int historyLength) {
        LogCollector logCollector = new LogCollector(mContext, CollectorType.Log, scheduledExecutorService, futureList, label, historyLength);
        collectors.add(logCollector);
        return logCollector;
    }

    public void resume() {
        if (!running.get()) {
            for (Collector collector: collectors) {
                collector.resume();
            }
            running.set(true);
        }
    }

    public void pause() {
        if (running.get()) {
            for (Collector collector: collectors) {
                collector.pause();
            }
            running.set(false);
        }
    }

    public void registerListener(CollectorListener listener) {
        for (Collector collector: collectors) {
            collector.registerListener(listener);
        }
    }

    public void unregisterListener(CollectorListener listener) {
        for (Collector collector: collectors) {
            collector.unregisterListener(listener);
        }
    }

    public Collector getCollector(CollectorType collectorType) {
        for (Collector collector: collectors) {
            if (collector.getType() == collectorType) {
                return collector;
            }
        }
        return null;
    }

    public List<Collector> getCollectors() {
        return collectors;
    }
}
