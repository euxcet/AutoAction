package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.SensorEvent;

import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;

import java.util.List;
import java.util.concurrent.ThreadPoolExecutor;

public abstract class BaseAction {

    protected Context mContext;

    protected ActionConfig config;
    protected RequestListener requestListener;
    protected List<ActionListener> actionListener;

    protected boolean isStarted = false;
    protected ThreadPoolExecutor threadPoolExecutor;

    public BaseAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener, ThreadPoolExecutor threadPoolExecutor) {
        this.mContext = context;
        this.config = config;
        this.requestListener = requestListener;
        this.actionListener = actionListener;
        this.threadPoolExecutor = threadPoolExecutor;
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
