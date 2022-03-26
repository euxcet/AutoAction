package com.hcifuture.contextactionlibrary.collect.trigger;

import android.content.Context;
import android.util.Log;
import android.util.Pair;

import com.hcifuture.contextactionlibrary.collect.collector.BluetoothCollector;
import com.hcifuture.contextactionlibrary.collect.collector.Collector;
import com.hcifuture.contextactionlibrary.collect.collector.CompleteIMUCollector;
import com.hcifuture.contextactionlibrary.collect.collector.NonIMUCollector;
import com.hcifuture.contextactionlibrary.collect.collector.WifiCollector;
import com.hcifuture.contextactionlibrary.collect.data.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

public abstract class Trigger {

    protected static final String TAG = "CollectorTrigger";

    private Context mContext;

    private final int samplingPeriod = 10000;

    private boolean paused;

    public enum CollectorType {
        Bluetooth,
        CompleteIMU,
        SampledIMU,
        NonIMU,
        Location,
        Weather,
        Wifi,
        All
    }

    protected List<Collector> collectors = new ArrayList<>();

    private void initializeAll() {
        String triggerFolder = getName();
        collectors.add(new BluetoothCollector(mContext, triggerFolder));
        collectors.add(new CompleteIMUCollector(mContext, triggerFolder, samplingPeriod, 1));
        collectors.add(new NonIMUCollector(mContext, triggerFolder));
        collectors.add(new WifiCollector(mContext, triggerFolder));
    }

    private void initialize(CollectorType type) {
        String triggerFolder = getName();
        switch (type) {
            case Bluetooth:
                collectors.add(new BluetoothCollector(mContext, triggerFolder));
                break;
            case CompleteIMU:
                collectors.add(new CompleteIMUCollector(mContext, triggerFolder, samplingPeriod, 1));
                break;
            case NonIMU:
                collectors.add(new NonIMUCollector(mContext, triggerFolder));
                break;
            case Wifi:
                collectors.add(new WifiCollector(mContext, triggerFolder));
                break;
            case All:
                initializeAll();
                break;
            default:
                break;
        }
        paused = false;
    }

    public Trigger(Context context, List<CollectorType> types) {
        this.mContext = context;
        try {
            for (CollectorType type : types) {
                initialize(type);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public Trigger(Context context, CollectorType type) {
        this.mContext = context;
        try {
            initialize(type);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public List<Pair<String, CompletableFuture<Data>>> triggerAsync() {
        return new LinkedList<>();
    }

    public abstract void trigger();

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
        Log.e("Trigger", "pause");
        for (Collector collector: collectors) {
            collector.pause();
        }
        paused = true;
    }

    public void resume() {
        Log.e("Trigger", "resume");
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
        Log.e("Trigger", "clean");
        for (Collector collector: collectors) {
            collector.cleanData();
        }
    }
}
