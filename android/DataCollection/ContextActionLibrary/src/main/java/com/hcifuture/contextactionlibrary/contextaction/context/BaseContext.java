package com.hcifuture.contextactionlibrary.contextaction.context;

import android.content.Context;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;

import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.contextactionlibrary.contextaction.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public abstract class BaseContext {
    protected Context mContext;

    protected ContextConfig config;

    protected RequestListener requestListener;
    protected List<ContextListener> contextListener;

    protected boolean isStarted = false;

    protected ScheduledExecutorService scheduledExecutorService;
    protected List<ScheduledFuture<?>> futureList;

    protected LogCollector logCollector;

    public BaseContext(Context context, ContextConfig config, RequestListener requestListener, List<ContextListener> contextListener, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        this.mContext = context;
        this.config = config;
        this.requestListener = requestListener;
        this.contextListener = contextListener;
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
        this.logCollector = null;
    }

    public void setLogCollector(LogCollector collector) {
        this.logCollector = collector;
    }

    public LogCollector getLogCollector() {
        return this.logCollector;
    }

    public abstract void start();
    public abstract void stop();
    // public abstract void onIMUSensorChanged(SensorEvent event);
    // public abstract void onProximitySensorChanged(SensorEvent event);
    public abstract void onIMUSensorEvent(SingleIMUData data);
    public abstract void onNonIMUSensorEvent(NonIMUData data);

    public abstract void onAccessibilityEvent(AccessibilityEvent event);
    public abstract void onBroadcastEvent(BroadcastEvent event);

    public abstract void getContext();

    public ContextConfig getConfig() {
        return config;
    }
}
