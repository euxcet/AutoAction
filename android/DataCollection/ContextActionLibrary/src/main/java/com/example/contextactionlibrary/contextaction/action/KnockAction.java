package com.example.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.util.Log;

import com.example.contextactionlibrary.model.NcnnInstance;
import com.example.ncnnlibrary.communicate.ActionConfig;
import com.example.ncnnlibrary.communicate.ActionListener;
import com.example.ncnnlibrary.communicate.ActionResult;

public class KnockAction extends ActionBase {

    private final int DATA_LENGTH = 128 * 6;
    private final int DATA_ELEMSIZE = 6;
    private final int INTERVAL = 9900000;

    private float[] data = new float[DATA_LENGTH];
    private long lastTimestampGyro = 0;
    private long lastTimestampLinear = 0;

    public KnockAction(Context context, ActionConfig config, ActionListener actionListener) {
        super(context, config, actionListener);
    }

    @Override
    public void start() {
        isStarted = true;
    }

    @Override
    public void stop() {
        isStarted = false;
    }

    @Override
    public void onAlwaysOnSensorChanged(SensorEvent event) {
        int type = event.sensor.getType();
        switch (type) {
            case Sensor.TYPE_GYROSCOPE:
                if (event.timestamp - lastTimestampGyro > INTERVAL) {
                    for (int i = 0; i < DATA_LENGTH - DATA_ELEMSIZE; i += DATA_ELEMSIZE) {
                        for (int j = 0; j < 3; j ++) {
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
                        for (int j = 3; j < 6; j ++) {
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
    }

    @Override
    public void getAction() {
        if (!isStarted)
            return;
        if (NcnnInstance.getInstance() != null) {
            int result = NcnnInstance.getInstance().actionDetect(data);
            if (result == 0) {
                actionListener.onAction(new ActionResult("Knock"));
            }
        }
    }
}
