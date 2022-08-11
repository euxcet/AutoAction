package com.hcifuture.datacollection.inference;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;

import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class ImuSensorManager implements SensorEventListener {

    private Context mContext;

    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor mGyroscope;
    private Sensor mLinear;
    private Sensor mMagnetic;

    private boolean isSensorOpened;
    private boolean isInitialized;
    private boolean isStarted;

    private final int DATA_LENGTH = 128 * 6;
    private final int DATA_ELEMSIZE = 6;
    private final int INTERVAL = 9900000;

    private float[] data = new float[DATA_LENGTH];
    private long lastTimestampGyro = 0;
    private long lastTimestampLinear = 0;

    private long lastKnockTimestamp = 0;

    private ThreadPoolExecutor threadPoolExecutor;

    private List<ImuEventListener> listeners;

    public ImuSensorManager(Context context) {
        this.mContext = context;
        threadPoolExecutor = new ThreadPoolExecutor(1, 2, 1000,
                TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(10),
                Executors.defaultThreadFactory(), new ThreadPoolExecutor.DiscardOldestPolicy());
        initialize();
    }

    public boolean initialize() {
        if (mSensorManager == null) {
            mSensorManager = (SensorManager) mContext.getSystemService(Context.SENSOR_SERVICE);
            mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
            mLinear = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
            mMagnetic = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        }

        isSensorOpened = true;
        isInitialized = true;

        return true;
    }

    public void addListener(ImuEventListener listener) {
        for (ImuEventListener l: listeners) {
            if (l == listener) {
                return;
            }
        }
        listeners.add(listener);
    }

    public void removeListener(ImuEventListener listener) {
        listeners.removeIf(l -> l == listener);
    }

    public void registerSensorListener() {
        if (isSensorOpened) {
            mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
            mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_FASTEST);
            mSensorManager.registerListener(this, mLinear, SensorManager.SENSOR_DELAY_FASTEST);
            mSensorManager.registerListener(this, mMagnetic, SensorManager.SENSOR_DELAY_FASTEST);
        }
    }

    public void unRegisterSensorListener() {
        if (isSensorOpened) {
            mSensorManager.unregisterListener(this, mAccelerometer);
            mSensorManager.unregisterListener(this, mGyroscope);
            mSensorManager.unregisterListener(this, mLinear);
            mSensorManager.unregisterListener(this, mMagnetic);
        }
    }

    public void start() {
        if (isStarted) {
            return;
        }
        if (!isInitialized) {
            if (!initialize()) {
                return;
            }
        }
        registerSensorListener();
        isStarted = true;
    }

    public void stop() {
        if (!isStarted) {
            return;
        }
        unRegisterSensorListener();
        isStarted = false;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        int type = event.sensor.getType();
        switch (type) {
            case Sensor.TYPE_GYROSCOPE:
                if (event.timestamp - lastTimestampGyro > INTERVAL) {
                    for (int i = 0; i < DATA_LENGTH - DATA_ELEMSIZE; i += DATA_ELEMSIZE) {
                        for (int j = 0; j < 3; j++) {
                            data[i + j] = data[i + j + DATA_ELEMSIZE];
                        }
                    }
                    data[DATA_LENGTH - DATA_ELEMSIZE] = event.values[0];
                    data[DATA_LENGTH - DATA_ELEMSIZE + 1] = event.values[1];
                    data[DATA_LENGTH - DATA_ELEMSIZE + 2] = event.values[2];
                    lastTimestampGyro = event.timestamp;
                }
                break;
            case Sensor.TYPE_LINEAR_ACCELERATION:
                if (event.timestamp - lastTimestampLinear > INTERVAL) {
                    for (int i = 0; i < DATA_LENGTH - DATA_ELEMSIZE; i += DATA_ELEMSIZE) {
                        for (int j = 3; j < 6; j++) {
                            data[i + j] = data[i + j + DATA_ELEMSIZE];
                        }
                    }
                    data[DATA_LENGTH - DATA_ELEMSIZE + 3] = event.values[0];
                    data[DATA_LENGTH - DATA_ELEMSIZE + 4] = event.values[1];
                    data[DATA_LENGTH - DATA_ELEMSIZE + 5] = event.values[2];
                    lastTimestampLinear = event.timestamp;
                }
                break;
            default:
                break;
        }

        // TODO: stable on hand, stable on table, not stable

        threadPoolExecutor.execute(() -> {
            float[] input_data = data.clone();
            if (isStarted) {
                if (Inferencer.getInstance() != null) {
                    int result = Inferencer.getInstance().inference("best.mnn", input_data);
                    // TODO: event
                }
            }
        });
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {}
}
