package com.example.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.SensorEvent;

public abstract class ActionBase {

    protected Context mContext;

    protected ActionListener actionListener;

    protected int seqLength;
    protected int classNum;
    protected String[] actions;

    protected boolean isStarted = false;

    public ActionBase(Context context, ActionListener actionListener, int seqLength, String[] actions) {
        this.mContext = context;
        this.actionListener = actionListener;
        this.seqLength = seqLength;
        this.classNum = actions.length;
        this.actions = actions;
    }

    public abstract void start();
    public abstract void stop();
    public abstract void onAlwaysOnSensorChanged(SensorEvent event);

    public abstract void getAction();
}
