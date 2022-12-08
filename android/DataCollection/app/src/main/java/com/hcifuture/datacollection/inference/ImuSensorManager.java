package com.hcifuture.datacollection.inference;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;
import android.util.Pair;

import com.hcifuture.datacollection.inference.filter.ImuFilter;

import org.checkerframework.checker.units.qual.A;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

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

    private final int DATA_SAMPLES = 100;
    private final int DATA_ELEMSIZE = 9;
    private final int INTERVAL = 9900000;

    private float[][] data = new float[DATA_ELEMSIZE][DATA_SAMPLES];

    private long lastTimestampAcc = 0;
    private long lastTimestampGyro = 0;
    private long lastTimestampLinear = 0;

    private long lastKnockTimestamp = 0;

    private ThreadPoolExecutor threadPoolExecutor;

    private List<ImuEventListener> listeners;

    private ImuFilter lowPassFilter;
    private ImuFilter bandPassFilter;
    private ImuFilter highPassFilter;
    private ImuFilter peakFilter;
    private AtomicInteger lock = new AtomicInteger(0);

    private long[] lastResultTimestamp = new long[10];
    private int[] resultCounter = new int[10];
    private long lastEventTimestamp = 0;

    public ImuSensorManager(Context context) {
        this.mContext = context;
        threadPoolExecutor = new ThreadPoolExecutor(1, 2, 1000,
                TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(10),
                Executors.defaultThreadFactory(), new ThreadPoolExecutor.DiscardOldestPolicy());
        initialize();
        float fs = 100.0f;
        float tw = 1.0f;
        float fc_low = 6.0f;
        float fc_high = 12.0f;
        float fc_peak = 0.5f;
        lowPassFilter = new ImuFilter("low-pass", fs, tw, fc_low, 0, "hamming");
        bandPassFilter = new ImuFilter("band-pass", fs, tw, fc_low, fc_high, "hamming");
        highPassFilter = new ImuFilter("high-pass", fs, tw, 0, fc_high, "hamming");
        peakFilter = new ImuFilter("low-pass", fs, tw, fc_peak, 0, "hamming");
        for (int i = 0; i < lastResultTimestamp.length; i++) {
            lastResultTimestamp[i] = 0;
            resultCounter[i] = 0;
        }
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
        listeners = new ArrayList<>();

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
            // acc linear gyro
            case Sensor.TYPE_ACCELEROMETER:
                if (event.timestamp - lastTimestampAcc > INTERVAL) {
                    for (int i = 0; i < 3; i++) {
                        System.arraycopy(data[i], 1, data[i], 0, DATA_SAMPLES - 1);
                    }
                    data[0][DATA_SAMPLES - 1] = event.values[0];
                    data[1][DATA_SAMPLES - 1] = event.values[1];
                    data[2][DATA_SAMPLES - 1] = event.values[2];
                    lastTimestampAcc = event.timestamp;
                }
                break;
            case Sensor.TYPE_LINEAR_ACCELERATION:
                if (event.timestamp - lastTimestampLinear > INTERVAL) {
                    for (int i = 3; i < 6; i++) {
                        System.arraycopy(data[i], 1, data[i], 0, DATA_SAMPLES - 1);
                    }
                    data[3][DATA_SAMPLES - 1] = event.values[0];
                    data[4][DATA_SAMPLES - 1] = event.values[1];
                    data[5][DATA_SAMPLES - 1] = event.values[2];
                    lastTimestampLinear = event.timestamp;
                }
                break;
            case Sensor.TYPE_GYROSCOPE:
                if (event.timestamp - lastTimestampGyro > INTERVAL) {
                    for (int i = 6; i < 9; i++) {
                        System.arraycopy(data[i], 1, data[i], 0, DATA_SAMPLES - 1);
                    }
                    data[6][DATA_SAMPLES - 1] = event.values[0];
                    data[7][DATA_SAMPLES - 1] = event.values[1];
                    data[8][DATA_SAMPLES - 1] = event.values[2];
                    lastTimestampGyro = event.timestamp;
                }
                break;
            default:
                break;
        }

        // TODO: stable on hand, stable on table, not stable

        threadPoolExecutor.execute(() -> {
            float[][] input_data = new float[DATA_ELEMSIZE * 4][DATA_SAMPLES];
            float[] norm = new float[DATA_SAMPLES];
            // acc_x acc_y acc_z linear_x linear_y linear_z gyro_x gyro_y gyro_z
            for (int i = 0; i < 9; i++) {
                input_data[i * 4] = data[i].clone();
            }
            for (int i = 0; i < DATA_SAMPLES; i++) {
                norm[i] = (float)Math.sqrt(Math.pow(input_data[3 * 4][i], 2.0)
                        + Math.pow(input_data[4 * 4][i], 2.0)
                        + Math.pow(input_data[5 * 4][i], 2.0));
            }
//            norm = peakFilter.filter(norm);
            int peak_pos = 0;
            float peak_v = -1000.0f;
            for (int i = 0; i < DATA_SAMPLES; i++) {
                if (norm[i] > peak_v) {
                    peak_v = norm[i];
                    peak_pos = i;
                }
            }
            if (peak_pos >= 55 && peak_pos <= 65 && peak_v > 6) {
                for (int i = 0; i < 9; i++) {
                    input_data[i * 4 + 1] = lowPassFilter.filter(input_data[i * 4]);
                    input_data[i * 4 + 2] = bandPassFilter.filter(input_data[i * 4]);
                    input_data[i * 4 + 3] = highPassFilter.filter(input_data[i * 4]);
                }
                float[] data_1d = new float[DATA_ELEMSIZE * 4 * DATA_SAMPLES];
                for (int j = 0; j < DATA_SAMPLES; j++) {
                    for (int i = 0; i < DATA_ELEMSIZE * 4; i++) {
                        data_1d[j * DATA_ELEMSIZE * 4 + i] = input_data[i][j];
                    }
                }
                if (isStarted) {
                    if (Inferencer.getInstance() != null) {
                        InferenceResult result = Inferencer.getInstance().inferenceAction("best.mnn", data_1d);
                        int id = result.classId;
                        String name = result.className;
                        if (!name.equals("negative")) {
                            long timestamp = System.currentTimeMillis();
                            for (int i = 0; i < lastResultTimestamp.length; i++) {
                                if (i != id) {
                                    lastResultTimestamp[i] = 0;
                                    resultCounter[i] = 0;
                                }
                            }
                            if (timestamp < lastResultTimestamp[id] + 100) {
                                resultCounter[id] += 1;
                            } else {
                                resultCounter[id] = 1;
                            }
                            lastResultTimestamp[id] = timestamp;
                            if (resultCounter[id] == 5) {
                                if (timestamp > lastEventTimestamp + 800) {
                                    lastEventTimestamp = timestamp;
                                    for (ImuEventListener listener: listeners) {
                                        listener.onAction(name);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {}
}
