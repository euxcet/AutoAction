package com.hcifuture.contextactionlibrary.collect.trigger;

import android.content.Context;
import android.util.Log;
import android.util.Pair;

import com.hcifuture.contextactionlibrary.collect.collector.BluetoothCollector;
import com.hcifuture.contextactionlibrary.collect.collector.Collector;
import com.hcifuture.contextactionlibrary.collect.collector.CompleteIMUCollector;
import com.hcifuture.contextactionlibrary.collect.collector.LocationCollector;
import com.hcifuture.contextactionlibrary.collect.collector.LogCollector;
import com.hcifuture.contextactionlibrary.collect.collector.NonIMUCollector;
import com.hcifuture.contextactionlibrary.collect.collector.WifiCollector;
import com.hcifuture.contextactionlibrary.collect.data.Data;
import com.hcifuture.shared.communicate.SensorType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public abstract class Trigger {

    protected static final String TAG = "CollectorTrigger";

    private Context mContext;

    private final int samplingPeriod = 10000;

    private boolean paused;

    protected ScheduledExecutorService scheduledExecutorService;
    protected List<ScheduledFuture<?>> futureList;

    public enum CollectorType {
        Bluetooth,
        CompleteIMU,
        SampledIMU,
        NonIMU,
        Location,
        Weather,
        Wifi,
        Log,
        All
    }

    protected List<Collector> collectors = new ArrayList<>();

    private void initializeAll() {
        String triggerFolder = getName();
        collectors.add(new BluetoothCollector(mContext, CollectorType.Bluetooth, triggerFolder, scheduledExecutorService, futureList));
        collectors.add(new CompleteIMUCollector(mContext, CollectorType.CompleteIMU, triggerFolder, scheduledExecutorService, futureList, samplingPeriod, 1));
        collectors.add(new NonIMUCollector(mContext, CollectorType.NonIMU, triggerFolder, scheduledExecutorService, futureList));
        collectors.add(new WifiCollector(mContext, CollectorType.Wifi, triggerFolder, scheduledExecutorService, futureList));
        collectors.add(new LocationCollector(mContext, CollectorType.Location, triggerFolder, scheduledExecutorService, futureList));
    }

    private void initialize(CollectorType type) {
        String triggerFolder = getName();
        switch (type) {
            case Bluetooth:
                collectors.add(new BluetoothCollector(mContext, CollectorType.Bluetooth, triggerFolder, scheduledExecutorService, futureList));
                break;
            case CompleteIMU:
                collectors.add(new CompleteIMUCollector(mContext, CollectorType.CompleteIMU, triggerFolder, scheduledExecutorService, futureList, samplingPeriod, 1));
                break;
            case NonIMU:
                collectors.add(new NonIMUCollector(mContext, CollectorType.NonIMU, triggerFolder, scheduledExecutorService, futureList));
                break;
            case Wifi:
                collectors.add(new WifiCollector(mContext, CollectorType.Wifi, triggerFolder, scheduledExecutorService, futureList));
                break;
            case Location:
                collectors.add(new LocationCollector(mContext, CollectorType.Location, triggerFolder, scheduledExecutorService, futureList));
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
        paused = false;
    }

    public Trigger(Context context, List<CollectorType> types, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
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
    }

    public Trigger(Context context, CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        this.mContext = context;
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
        try {
            initialize(type);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public List<Pair<String, CompletableFuture<Data>>> triggerAsync() {
        return new LinkedList<>();
    }

    public abstract CompletableFuture<Void> trigger();

    public abstract CompletableFuture<Void> trigger(List<CollectorType> types);

    public abstract CompletableFuture<Void> trigger(Collector collector);

    public abstract String getName();

    public void close() {
        for (Collector collector: collectors) {
            collector.close();
        }
    }

    public synchronized Map<String, Data> getData() {
        Map<String, Data> result = new HashMap<>();
        for (Collector collector: collectors) {
            if (collector.forPrediction()) {
                result.put(collector.getSaveFolderName(), collector.getData());
            }
        }
        return result;
    }

    public void pause() {
        for (Collector collector: collectors) {
            collector.pause();
        }
        paused = true;
    }

    public void resume() {
        for (Collector collector: collectors) {
            collector.resume();
        }
        paused = false;
    }

    public boolean isPaused() {
        return paused;
    }

    public List<Collector> getCollectors() {
        return collectors;
    }

    public void cleanData() {
        for (Collector collector: collectors) {
            collector.cleanData();
        }
    }

    public LogCollector newLogCollector(String label, int historyLength) {
        String triggerFolder = getName();
        LogCollector logCollector = new LogCollector(mContext, CollectorType.Log, triggerFolder, scheduledExecutorService, futureList, label, historyLength);
        collectors.add(logCollector);
        return logCollector;
    }

    public String getRecentPath(CollectorType type) {
        for (Collector collector: collectors) {
            if (collector.getType() == type) {
                return collector.getRecentPath();
            }
        }
        return null;
    }
}
