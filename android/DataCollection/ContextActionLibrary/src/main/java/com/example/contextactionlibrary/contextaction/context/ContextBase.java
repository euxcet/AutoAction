package com.example.contextactionlibrary.contextaction.context;

import android.content.Context;
import android.hardware.SensorEvent;

import com.example.ncnnlibrary.communicate.config.ContextConfig;
import com.example.ncnnlibrary.communicate.listener.ContextListener;

import java.util.List;

public abstract class ContextBase {

    protected Context mContext;

    protected ContextConfig config;
    protected List<ContextListener> contextListener;

    protected boolean isStarted = false;

    public ContextBase(Context context, ContextConfig config, List<ContextListener> contextListener) {
        this.mContext = context;
        this.config = config;
        this.contextListener = contextListener;
    }

    public abstract void start();
    public abstract void stop();
    public abstract void onIMUSensorChanged(SensorEvent event);
    public abstract void onProximitySensorChanged(SensorEvent event);

    public abstract void getContext();

    public ContextConfig getConfig() {
        return config;
    }
}
