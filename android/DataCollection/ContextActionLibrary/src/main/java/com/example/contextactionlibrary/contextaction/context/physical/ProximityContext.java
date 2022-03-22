package com.example.contextactionlibrary.contextaction.context.physical;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;

import com.example.contextactionlibrary.contextaction.context.BaseContext;
import com.example.ncnnlibrary.communicate.config.ContextConfig;
import com.example.ncnnlibrary.communicate.event.BroadcastEvent;
import com.example.ncnnlibrary.communicate.listener.ContextListener;
import com.example.ncnnlibrary.communicate.listener.RequestListener;
import com.example.ncnnlibrary.communicate.result.ContextResult;

import java.util.List;

public class ProximityContext extends BaseContext {
    private String TAG = "ProximityContext";

    private long lastTimestamp = -1;

    private static int proxThreshold = 4;
    private long lastNearTime = 0L;
    private long lastFarTime = 0L;
    private long lastRecognized = 0L;

    public ProximityContext(Context context, ContextConfig config, RequestListener requestListener, List<ContextListener> contextListener) {
        super(context, config, requestListener, contextListener);
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
        if (event.sensor.getType() != Sensor.TYPE_PROXIMITY)
            return;
        float prox = event.values[0];
        if ((int)prox < proxThreshold) {
            lastNearTime = event.timestamp;
        }
        else {
            lastFarTime = event.timestamp;
        }
        lastTimestamp = event.timestamp;
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {

    }

    @Override
    public void onBroadcastEvent(BroadcastEvent event) {

    }

    @Override
    public void getContext() {
        if (!isStarted) {
            return;
        }
        if (lastTimestamp - lastNearTime < 1.5 * 1e9 && lastTimestamp - lastRecognized > 5 * 1e9) {
            lastRecognized = lastTimestamp;
            if (contextListener != null) {
                for (ContextListener listener: contextListener) {
                    listener.onContext(new ContextResult("Proximity"));
                }
            }
        }
    }
}
