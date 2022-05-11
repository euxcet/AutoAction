package com.hcifuture.contextactionlibrary.sensor.collector;

import android.content.Context;
import android.util.Log;

import com.hcifuture.contextactionlibrary.sensor.collector.async.AudioCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.BluetoothCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.GPSCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.IMUCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.LocationCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.NonIMUCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.WifiCollector;
import com.hcifuture.shared.communicate.listener.RequestListener;

import org.checkerframework.checker.units.qual.C;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.atomic.AtomicBoolean;

public class CollectorManager {
    private Context mContext;
    private ScheduledExecutorService scheduledExecutorService;
    private List<ScheduledFuture<?>> futureList;
    private RequestListener requestListener;

    private AtomicBoolean running = new AtomicBoolean(false);
    private List<Collector> collectors = new ArrayList<>();

    public enum CollectorType {
        Audio,
        Bluetooth,
        IMU,
        NonIMU,
        Location,
        Weather,
        GPS,
        Wifi,
        Log,
        All
    }

    private void initializeAll() {
        initialize(CollectorType.Bluetooth);
        initialize(CollectorType.IMU);
        initialize(CollectorType.NonIMU);
        initialize(CollectorType.Wifi);
        initialize(CollectorType.Location);
        initialize(CollectorType.Audio);
        initialize(CollectorType.GPS);
    }

    private void initialize(CollectorType type) {
        Collector collector = null;
        switch (type) {
            case Bluetooth:
                collector = new BluetoothCollector(mContext, CollectorType.Bluetooth, scheduledExecutorService, futureList);
                break;
            case IMU:
                collector = new IMUCollector(mContext, CollectorType.IMU, scheduledExecutorService, futureList);
                break;
            case NonIMU:
                collector = new NonIMUCollector(mContext, CollectorType.NonIMU, scheduledExecutorService, futureList);
                break;
            case Wifi:
                collector = new WifiCollector(mContext, CollectorType.Wifi, scheduledExecutorService, futureList);
                break;
            case Location:
                collector = new LocationCollector(mContext, CollectorType.Location, scheduledExecutorService, futureList);
                break;
            case Audio:
                collector = new AudioCollector(mContext, CollectorType.Audio, scheduledExecutorService, futureList);
                break;
            case GPS:
                collector = new GPSCollector(mContext, CollectorType.GPS, scheduledExecutorService, futureList);
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
        if (collector != null) {
            collector.setRequestListener(requestListener);
            collectors.add(collector);
        }
    }

    public CollectorManager(Context context, List<CollectorType> types, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener) {
        this.mContext = context;
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
        this.requestListener = requestListener;
        try {
            for (CollectorType type : types) {
                initialize(type);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        running.set(true);
    }

    public CollectorManager(Context context, CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener) {
        this.mContext = context;
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
        this.requestListener = requestListener;
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
