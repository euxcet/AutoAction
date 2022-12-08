package com.hcifuture.datacollection.inference;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import com.hcifuture.datacollection.inference.filter.ImuFilter;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class OrientationManager implements SensorEventListener {
    private Context mContext;
    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor mMagnetometer;

    private boolean isStarted;

    private float[] mGravity;
    private float[] mMagnetic;

    public OrientationManager(Context context) {
        this.mContext = context;
        this.mSensorManager = (SensorManager)context.getSystemService(Context.SENSOR_SERVICE);
        this.mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        this.mMagnetometer = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        this.isStarted = false;
    }

    private void registerSensorListener() {
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mMagnetometer, SensorManager.SENSOR_DELAY_FASTEST);
    }

    private void unregisterSensorListener() {
        mSensorManager.unregisterListener(this);
    }

    public void start() {
        if (isStarted) {
            return;
        }
        registerSensorListener();
        isStarted = true;
    }

    public void stop() {
        if (!isStarted) {
            return;
        }
        unregisterSensorListener();
        isStarted = false;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            mGravity = event.values;
        }
        if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD) {
            mMagnetic = event.values;
        }
        if (mGravity != null && mMagnetic != null) {
            float R[] = new float[9];
            float I[] = new float[9];
            boolean result = SensorManager.getRotationMatrix(R, I, mGravity, mMagnetic);
            if (result) {
                float orientation[] = new float[3];
                SensorManager.getOrientation(R, orientation);
                Log.e("Orientation", String.format("%.2f %.2f %.2f", orientation[0], orientation[1], orientation[2]));
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
