package com.example.datacollection.contextaction.sensor;

import java.lang.reflect.Method;
import java.util.Timer;
import java.util.TimerTask;

public abstract class MySensorManager {

    protected String TAG = "MySensorManager";
    protected String name;
    protected Object container;
    protected Method onSensorChanged;

    protected boolean isInitialized = false;
    protected boolean isStarted = false;
    protected boolean isSensorOpened = false;

    public void setName(String name) {
        this.name = name;
    }
    public String getName() {
        return name;
    };

    public Object getContainer() {
        return container;
    }

    public Method getOnSensorChanged() {
        return onSensorChanged;
    }

    public void setContainer(Object container) {
        this.container = container;
    }

    public void setOnSensorChanged(Method onSensorChanged) {
        this.onSensorChanged = onSensorChanged;
    }

    public abstract void start();
    public abstract void stop();
    public void stopLater(long millisecond) {
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                stop();
            }
        }, millisecond);
    };
}
