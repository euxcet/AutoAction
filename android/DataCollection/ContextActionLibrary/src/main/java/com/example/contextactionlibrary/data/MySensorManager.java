package com.example.contextactionlibrary.data;

import java.util.Timer;
import java.util.TimerTask;

public abstract class MySensorManager {

    protected String TAG = "MySensorManager";
    protected String name;

    protected boolean isInitialized = false;
    protected boolean isStarted = false;
    protected boolean isSensorOpened = false;

    public void setName(String name) {
        this.name = name;
    }
    public String getName() {
        return name;
    };

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
