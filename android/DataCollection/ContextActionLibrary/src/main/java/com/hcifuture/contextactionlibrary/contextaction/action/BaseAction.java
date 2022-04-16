package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.SensorEvent;

import com.hcifuture.contextactionlibrary.sensor.data.Data;
import com.hcifuture.contextactionlibrary.sensor.data.IMUData;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;

import java.util.List;

public abstract class BaseAction {

    protected Context mContext;

    protected ActionConfig config;
    protected RequestListener requestListener;
    protected List<ActionListener> actionListener;

    protected boolean isStarted = false;

    public BaseAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener) {
        this.mContext = context;
        this.config = config;
        this.requestListener = requestListener;
        this.actionListener = actionListener;
    }

    public abstract void start();
    public abstract void stop();
    // public abstract void onIMUSensorChanged(SensorEvent event);
    // public abstract void onProximitySensorChanged(SensorEvent event);
    public abstract void onIMUSensorEvent(SingleIMUData data);
    public abstract void onNonIMUSensorEvent(NonIMUData data);

    public abstract void getAction();

    public ActionConfig getConfig() {
        return config;
    }
}
