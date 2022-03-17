package com.example.contextactionlibrary.contextaction.context;

import android.content.Context;
import android.hardware.SensorEvent;
import android.view.accessibility.AccessibilityEvent;

import com.example.ncnnlibrary.communicate.config.ContextConfig;
import com.example.ncnnlibrary.communicate.event.BroadcastEvent;
import com.example.ncnnlibrary.communicate.listener.ContextListener;
import com.example.ncnnlibrary.communicate.listener.RequestListener;

import java.util.List;

public abstract class ContextBase {
    protected Context mContext;

    protected ContextConfig config;

    protected RequestListener requestListener;
    protected List<ContextListener> contextListener;

    protected boolean isStarted = false;

    public ContextBase(Context context, ContextConfig config, RequestListener requestListener, List<ContextListener> contextListener) {
        this.mContext = context;
        this.config = config;
        this.requestListener = requestListener;
        this.contextListener = contextListener;
    }

    public abstract void start();
    public abstract void stop();
    public abstract void onIMUSensorChanged(SensorEvent event);
    public abstract void onProximitySensorChanged(SensorEvent event);

    public abstract void onAccessibilityEvent(AccessibilityEvent event);
    public abstract void onBroadcastEvent(BroadcastEvent event);

    public abstract void getContext();

    public ContextConfig getConfig() {
        return config;
    }
}
