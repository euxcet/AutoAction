package com.hcifuture.contextactionlibrary.sensor.collector;

import android.content.Context;

import com.hcifuture.shared.communicate.config.RequestConfig;
import com.hcifuture.shared.communicate.listener.RequestListener;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.atomic.AtomicBoolean;

public abstract class Collector {
    protected Context mContext;
    protected ScheduledExecutorService scheduledExecutorService;
    protected List<ScheduledFuture<?>> futureList;
    protected CollectorManager.CollectorType type;
    protected final List<CollectorListener> listenerList = new ArrayList<>();
    protected AtomicBoolean isRegistered = new AtomicBoolean(false);
    protected RequestListener requestListener = null;

    public Collector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        this.mContext = context;
        this.type = type;
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
        initialize();
    }

    public void setRequestListener(RequestListener requestListener) {
        this.requestListener = requestListener;
    }

    protected void notifyWake() {
        if (requestListener != null) {
            RequestConfig requestConfig = new RequestConfig();
            requestConfig.putValue("wake", true);
            requestListener.onRequest(requestConfig);
        }
    }

    public abstract void initialize();

    public abstract void close();

    public abstract void pause();

    public abstract void resume();

    public abstract String getName();

    public abstract String getExt();

    public void registerListener(CollectorListener listener) {
        synchronized (listenerList) {
            listenerList.add(listener);
        }
    }

    public void unregisterListener(CollectorListener listener) {
        synchronized (listenerList) {
            listenerList.remove(listener);
        }
    }

    public CollectorManager.CollectorType getType() {
        return type;
    }
}
