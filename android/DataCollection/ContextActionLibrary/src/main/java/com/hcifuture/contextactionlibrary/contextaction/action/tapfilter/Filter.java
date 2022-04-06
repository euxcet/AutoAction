package com.hcifuture.contextactionlibrary.contextaction.action.tapfilter;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

public abstract class Filter {
    protected final long MILLISECOND = (long)1e6;
    protected final long SECOND = (long)1e9;

    private final float[] accMark = new float[3];
    private final float[] magMark = new float[3];

    private final float[] rotationMatrix = new float[9];
    private final float[] orientationAngles = new float[3];

    protected long[] lastTime = new long[2];

    private final int ORIENTATION_CHECK_NUMBER = 10;

    private float[][] orientationMark = new float[ORIENTATION_CHECK_NUMBER][3];

    private final int SEQ_LENGTH = 10;
    private float[][] modelInput = new float[SEQ_LENGTH][6];

    private long[] lastSensorTime = new long[30];

    // just for horizontal / static cases' record && upload
    protected int linearStaticCount = 0;
    protected int gyroStaticCount = 0;

    public Filter() {

    }

    public abstract int passWithDelay(long timestamp);
    public abstract boolean passDirectly();


    private void updateOrientationAngles() {
        SensorManager.getRotationMatrix(rotationMatrix, null, accMark, magMark);
        SensorManager.getOrientation(rotationMatrix, orientationAngles);
        // orientationAngles: 方位角，俯仰角，倾侧角
        for (int i = 0; i < ORIENTATION_CHECK_NUMBER - 1; i++)
            System.arraycopy(orientationMark[i + 1], 0, orientationMark[i], 0, 3);
        System.arraycopy(orientationAngles, 0, orientationMark[ORIENTATION_CHECK_NUMBER - 1], 0, 3);
    }

    private void updateInput(int type, float x, float y, float z, long timestamp) {
        int idx = type == Sensor.TYPE_GYROSCOPE ? 0 : 1;
        if (timestamp < lastTime[idx] + 3 * MILLISECOND) {
            return;
        }
        lastTime[idx] = timestamp;
        for (int i = 0; i < SEQ_LENGTH - 1; i++) {
            System.arraycopy(modelInput[i + 1], 3 * idx, modelInput[i], 3 * idx, 3);
        }
        modelInput[SEQ_LENGTH - 1][3 * idx] = x;
        modelInput[SEQ_LENGTH - 1][3 * idx + 1] = y;
        modelInput[SEQ_LENGTH - 1][3 * idx + 2] = z;
    }

    public void feedSensorData(int type, float value0, float value1, float value2, long timestamp) {
        if (timestamp < lastSensorTime[type] + 8 * MILLISECOND) {
            return;
        }
        lastSensorTime[type] = timestamp;
        switch (type) {
            case Sensor.TYPE_GYROSCOPE:
            case Sensor.TYPE_LINEAR_ACCELERATION:
                updateInput(type, value0, value1, value2, timestamp);
                break;
            case Sensor.TYPE_ACCELEROMETER:
                accMark[0] = value0;
                accMark[1] = value1;
                accMark[2] = value2;
                break;
            case Sensor.TYPE_MAGNETIC_FIELD:
                magMark[0] = value0;
                magMark[1] = value1;
                magMark[2] = value2;
                updateOrientationAngles();
                break;
            default:
                break;
        }
    }

    public void onSensorChanged(SensorEvent event) {
        int type = event.sensor.getType();
        if (event.timestamp < lastSensorTime[type] + 8 * MILLISECOND) {
            return;
        }
        lastSensorTime[type] = event.timestamp;
        switch (type) {
            case Sensor.TYPE_GYROSCOPE:
            case Sensor.TYPE_LINEAR_ACCELERATION:
                updateInput(type, event.values[0], event.values[1], event.values[2], event.timestamp);
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

        // just for horizontal / static cases' record && upload
        float linearAccThreshold = 0.05f;
        float gyroThreshold = 0.02f;
        // linear
        if (event.sensor.getType() == 10) {
            if (Math.abs(event.values[0]) < linearAccThreshold && Math.abs(event.values[1]) < linearAccThreshold)
                linearStaticCount = Math.min(100, linearStaticCount + 1);
            else
                linearStaticCount = Math.max(0, linearStaticCount - 1);
        }
        // gyro
        else if (event.sensor.getType() == 4) {
            if (Math.abs(event.values[0]) < gyroThreshold && Math.abs(event.values[1]) < gyroThreshold && Math.abs(event.values[2]) < gyroThreshold)
                gyroStaticCount = Math.min(200, gyroStaticCount + 1);
            else
                gyroStaticCount = Math.max(0, gyroStaticCount - 1);
        }
    }

    protected boolean checkIsHorizontal(int yAllowed, int zAllowed) {
        int yZeroNum = 0, zZeroNum = 0;
        for (int i = 0; i < ORIENTATION_CHECK_NUMBER; i++) {
            if (Math.abs(orientationMark[i][1]) < 0.6)
                yZeroNum++;
            if (Math.abs(orientationMark[i][2]) < 0.3)
                zZeroNum++;
        }
        return yZeroNum >= ORIENTATION_CHECK_NUMBER - yAllowed && zZeroNum >= ORIENTATION_CHECK_NUMBER - zAllowed;
    }

    protected boolean checkIsStatic() {
        // 0.01 && 0.1
//        float gyroThreshold = 0.5f;
        float linearAccThreshold = 1.0f;
        for (int i = 0; i < SEQ_LENGTH; i++) {
//            for (int j = 0; j < 3; j++) {
//                if (Math.abs(modelInput[i][j]) > gyroThreshold) {
//                    return false;
//                }
//            }
            for (int j = 3; j < 6; j++) {
                if (Math.abs(modelInput[i][j]) > linearAccThreshold) {
                    return false;
                }
            }
        }
        return true;
    }
}
