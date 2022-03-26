package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;

import com.hcifuture.contextactionlibrary.model.NcnnInstance;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;

import java.util.List;

public class KnockAction extends BaseAction {

    private final int DATA_LENGTH = 128 * 6;
    private final int DATA_ELEMSIZE = 6;
    private final int INTERVAL = 9900000;

    private float[] data = new float[DATA_LENGTH];
    private long lastTimestampGyro = 0;
    private long lastTimestampLinear = 0;

    public KnockAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener) {
        super(context, config, requestListener, actionListener);
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
    public void onIMUSensorChanged(SensorEvent event) {
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
    public void onProximitySensorChanged(SensorEvent event) {

    }

    @Override
    public void getAction() {
        if (!isStarted)
            return;
        if (NcnnInstance.getInstance() != null) {
            int result = NcnnInstance.getInstance().actionDetect(data);
            if (result == 0) {
                for (ActionListener listener: actionListener) {
                    listener.onAction(new ActionResult("Knock"));
                }
            }
        }
    }
}
