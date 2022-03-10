package com.example.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.SensorEvent;

import com.example.ncnnlibrary.communicate.config.ActionConfig;
import com.example.ncnnlibrary.communicate.listener.ActionListener;

public abstract class ActionBase {

    protected Context mContext;

    protected ActionConfig config;
    protected ActionListener actionListener;

    protected boolean isStarted = false;

    public ActionBase(Context context, ActionConfig config, ActionListener actionListener) {
        this.mContext = context;
        this.actionListener = actionListener;
    }

    public abstract void start();
    public abstract void stop();
    public abstract void onIMUSensorChanged(SensorEvent event);
    public abstract void onProximitySensorChanged(SensorEvent event);

    public abstract void getAction();

    public ActionConfig getConfig() {
        return config;
    }
}
