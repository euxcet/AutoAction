package com.example.contextactionlibrary.contextaction.context;

import android.content.Context;
import android.hardware.SensorEvent;
import android.util.Log;

import com.example.contextactionlibrary.data.Preprocess;
import com.example.ncnnlibrary.communicate.config.ContextConfig;
import com.example.ncnnlibrary.communicate.listener.ContextListener;
import com.example.ncnnlibrary.communicate.result.ContextResult;

public class ProximityContext extends ContextBase {
    private String TAG = "ProximityContext";

    private Preprocess preprocess;

    private long lastRecognized = 0L;

    public ProximityContext(Context context, ContextConfig config, ContextListener contextListener) {
        super(context, config, contextListener);
        preprocess = Preprocess.getInstance();
    }

    @Override
    public synchronized void start() {
        if (isStarted) {
            Log.d(TAG, "Context is already started.");
            return;
        }
        isStarted = true;
    }

    @Override
    public synchronized void stop() {
        if (!isStarted) {
            Log.d(TAG, "Context is already stopped");
            return;
        }
        isStarted = false;
    }

    @Override
    public void onIMUSensorChanged(SensorEvent event) {

    }

    @Override
    public void onProximitySensorChanged(SensorEvent event) {
        // TODO: Split the logic of Preprocess here
    }

    @Override
    public void getContext() {
        if (!isStarted) {
            return;
        }
        long tmp = preprocess.checkLastNear((long)(1.5 * 1e9), lastRecognized);
        if (tmp != -1) {
            lastRecognized = tmp;
            if (contextListener != null) {
                contextListener.onContext(new ContextResult("Proximity"));
            }
        }
    }
}
