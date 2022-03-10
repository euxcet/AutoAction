package com.example.contextactionlibrary.data;

import android.content.Context;
import android.hardware.SensorEvent;

import com.example.contextactionlibrary.contextaction.action.ActionBase;
import com.example.contextactionlibrary.contextaction.context.ContextBase;

import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

public abstract class MySensorManager {
    protected String TAG = "MySensorManager";
    protected String name;

    protected Context mContext;
    protected boolean isInitialized = false;
    protected boolean isStarted = false;
    protected boolean isSensorOpened = false;

    protected List<ActionBase> actions;
    protected List<ContextBase> contexts;

    protected MySensorManager(Context context, String name, List<ActionBase> actions, List<ContextBase> contexts) {
        this.mContext = context;
        this.name = name;
        this.actions = actions;
        this.contexts = contexts;
    }

    public void setName(String name) {
        this.name = name;
    }
    public String getName() {
        return name;
    };

    public abstract void start();
    public abstract void stop();

    public abstract void onSensorChangedDex(SensorEvent event);

    public void stopLater(long millisecond) {
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                stop();
            }
        }, millisecond);
    };

}
