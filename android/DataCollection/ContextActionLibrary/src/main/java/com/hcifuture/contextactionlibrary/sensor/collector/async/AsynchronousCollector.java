package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.content.Context;

import com.google.gson.Gson;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.data.Data;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public abstract class AsynchronousCollector extends Collector {
    protected Gson gson = new Gson();

    public AsynchronousCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
    }

    public abstract CompletableFuture<Data> getData(TriggerConfig config);

    public abstract CompletableFuture<String> getDataString(TriggerConfig config);
}
