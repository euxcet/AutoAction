package com.hcifuture.contextactionlibrary.sensor.collector.sync;

import android.content.Context;

import com.google.gson.Gson;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.data.Data;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public abstract class SynchronousCollector extends Collector {
    protected Gson gson = new Gson();

    public SynchronousCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
    }

    public abstract Data getData(TriggerConfig config);

    public abstract String getDataString(TriggerConfig config);
}
