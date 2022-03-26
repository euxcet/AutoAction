package com.hcifuture.contextactionlibrary.contextaction.context.physical;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;

import com.hcifuture.contextactionlibrary.contextaction.context.BaseContext;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.shared.communicate.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.util.List;

public class TableContext extends BaseContext {

    private String TAG = "TableContext";

    private final float[] accMark = new float[3];
    private final float[] magMark = new float[3];
    private final float[] rotationMatrix = new float[9];
    private final float[] orientationAngles = new float[3];
    private final int ORIENTATION_CHECK_NUMBER = 10;
    private final float[][] orientationMark = new float[ORIENTATION_CHECK_NUMBER][3];

    private int linearStaticCount = 0;
    private int gyroStaticCount = 0;

    private boolean isOnTable = false;

    public TableContext(Context context, ContextConfig config, RequestListener requestListener, List<ContextListener> contextListener) {
        super(context, config, requestListener, contextListener);
    }

    private void checkIsStatic(SensorEvent event) {
        float linearAccThreshold = 0.05f;
        float gyroThreshold = 0.02f;
        // linear acc
        if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            if (Math.abs(event.values[0]) <= linearAccThreshold && Math.abs(event.values[1]) <= linearAccThreshold)
                linearStaticCount = Math.min(20, linearStaticCount + 1);
            else
                linearStaticCount = Math.max(0, linearStaticCount - 1);
        }
        // gyro
        else {
            if (Math.abs(event.values[0]) <= gyroThreshold && Math.abs(event.values[1]) <= gyroThreshold && Math.abs(event.values[2]) <= gyroThreshold)
                gyroStaticCount = Math.min(40, gyroStaticCount + 1);
            else
                gyroStaticCount = Math.max(0, gyroStaticCount - 1);
        }
    }

    protected boolean checkIsHorizontal() {
        for (int i = 0; i < ORIENTATION_CHECK_NUMBER; i++)
            if (Math.abs(orientationMark[i][1]) > 0.1 || Math.abs(orientationMark[i][2]) > 0.1)
                return false;
        return true;
    }

    private void updateOrientationAngles() {
        SensorManager.getRotationMatrix(rotationMatrix, null, accMark, magMark);
        SensorManager.getOrientation(rotationMatrix, orientationAngles);
        for (int i = 0; i < ORIENTATION_CHECK_NUMBER - 1; i++)
            System.arraycopy(orientationMark[i + 1], 0, orientationMark[i], 0, 3);
        System.arraycopy(orientationAngles, 0, orientationMark[ORIENTATION_CHECK_NUMBER - 1], 0, 3);
    }

    @Override
    public synchronized void start() {
        if (isStarted) {
            Log.d(TAG, "Context is already started.");
            return;
        }
        isStarted = true;
        isOnTable = false;
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
        switch (event.sensor.getType()) {
            case Sensor.TYPE_GYROSCOPE:
            case Sensor.TYPE_LINEAR_ACCELERATION:
                checkIsStatic(event);
                break;
            case Sensor.TYPE_ACCELEROMETER:
                System.arraycopy(event.values, 0, accMark, 0, accMark.length);
                break;
            case Sensor.TYPE_MAGNETIC_FIELD:
                System.arraycopy(event.values, 0, magMark, 0, magMark.length);
                updateOrientationAngles();
                break;
            default:
                break;
        }
    }

    @Override
    public void onProximitySensorChanged(SensorEvent event) {

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
        if (linearStaticCount > 10 && gyroStaticCount > 20 && checkIsHorizontal()) {
            if (isOnTable)
                return;
            isOnTable = true;
            if (contextListener != null) {
                for (ContextListener listener: contextListener) {
                    listener.onContext(new ContextResult("Table"));
                }
            }
        }
        else {
            if (!isOnTable)
                return;
            isOnTable = false;
            if (contextListener != null) {
                for (ContextListener listener: contextListener) {
                    listener.onContext(new ContextResult("Not on Table"));
                }
            }
        }
    }
}
