package com.hcifuture.contextactionlibrary.sensor.trigger;

import android.content.Context;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public abstract class Trigger {
    protected static final String TAG = "Trigger";

    protected Context mContext;
    protected CollectorManager collectorManager;
    protected ScheduledExecutorService scheduledExecutorService;
    protected List<ScheduledFuture<?>> futureList;

    public Trigger(Context context, CollectorManager collectorManager, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        this.mContext = context;
        this.collectorManager = collectorManager;
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
    }

    public abstract CompletableFuture<?> trigger(TriggerConfig config);

    public abstract CompletableFuture<?> trigger(List<CollectorManager.CollectorType> types, TriggerConfig config);

    public abstract CompletableFuture<?> trigger(Collector collector, TriggerConfig config);

    public Collector getCollector(CollectorManager.CollectorType collectorType) {
        return collectorManager.getCollector(collectorType);
    }

    /*
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
     */

    /*
    public String getRecentPath(CollectorType type) {
        for (Collector collector: collectors) {
            if (collector.getType() == type) {
                return collector.getRecentPath();
            }
        }
        return null;
    }
     */
}
