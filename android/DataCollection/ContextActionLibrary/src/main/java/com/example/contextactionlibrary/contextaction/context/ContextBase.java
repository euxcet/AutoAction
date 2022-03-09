package com.example.contextactionlibrary.contextaction.context;

import android.content.Context;

public abstract class ContextBase {

    protected Context mContext;

    protected ContextListener contextListener;

    protected int seqLength;
    protected int classNum;
    protected String[] contexts;

    protected boolean isStarted = false;

    public ContextBase(Context context, ContextListener contextListener, int seqLength, String[] contexts) {
        this.mContext = context;
        this.contextListener = contextListener;
        this.seqLength = seqLength;
        this.classNum = contexts.length;
        this.contexts = contexts;
    }

    public abstract void start();
    public abstract void stop();
    public abstract void getContext();
}
