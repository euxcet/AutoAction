package com.hcifuture.contextactionlibrary.collect.collector;

import android.content.Context;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.HandlerThread;

import com.hcifuture.contextactionlibrary.collect.trigger.Trigger;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public abstract class SensorCollector extends Collector {

    // protected HandlerThread sensorThread;
    protected Handler sensorHandler;
    protected SensorManager sensorManager;

    public SensorCollector(Context context, Trigger.CollectorType type, String triggerFolder, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, triggerFolder, scheduledExecutorService, futureList);
    }

    public abstract void addSensorData(float x, float y, float z, int idx, long time);
}
