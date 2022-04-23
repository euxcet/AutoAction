package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.Sensor;

import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class KnockAction extends BaseAction {

    private final int DATA_LENGTH = 128 * 6;
    private final int DATA_ELEMSIZE = 6;
    private final int INTERVAL = 9900000;

    private float[] inputData = new float[DATA_LENGTH];
    private long lastTimestampGyro = 0;
    private long lastTimestampLinear = 0;

    public KnockAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, config, requestListener, actionListener, scheduledExecutorService, futureList);
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
    public void onIMUSensorEvent(SingleIMUData data) {
        int type = data.getType();
        switch (type) {
            case Sensor.TYPE_GYROSCOPE:
                if (data.getTimestamp() - lastTimestampGyro > INTERVAL) {
                    for (int i = 0; i < DATA_LENGTH - DATA_ELEMSIZE; i += DATA_ELEMSIZE) {
                        for (int j = 0; j < 3; j ++) {
                            inputData[i + j] = inputData[i + j + DATA_ELEMSIZE];
                        }
                    }
                    inputData[DATA_LENGTH - DATA_ELEMSIZE] = data.getValues().get(0);
                    inputData[DATA_LENGTH - DATA_ELEMSIZE + 1] = data.getValues().get(1);
                    inputData[DATA_LENGTH - DATA_ELEMSIZE + 2] = data.getValues().get(2);
                    lastTimestampGyro = data.getTimestamp();
                }
                break;
            case Sensor.TYPE_LINEAR_ACCELERATION:
                if (data.getTimestamp() - lastTimestampLinear > INTERVAL) {
                    for (int i = 0; i < DATA_LENGTH - DATA_ELEMSIZE; i += DATA_ELEMSIZE) {
                        for (int j = 3; j < 6; j ++) {
                            inputData[i + j] = inputData[i + j + DATA_ELEMSIZE];
                        }
                    }
                    inputData[DATA_LENGTH - DATA_ELEMSIZE + 3] = data.getValues().get(0);
                    inputData[DATA_LENGTH - DATA_ELEMSIZE + 4] = data.getValues().get(1);
                    inputData[DATA_LENGTH - DATA_ELEMSIZE + 5] = data.getValues().get(2);
                    lastTimestampLinear = data.getTimestamp();
                }
                break;
            default:
                break;
        }
    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {

    }

    @Override
    public void getAction() {
        if (!isStarted) {
            return;
        }
        /*
        if (NcnnInstance.getInstance() != null) {
            int result = NcnnInstance.getInstance().actionDetect(inputData);
            if (result == 0) {
                for (ActionListener listener: actionListener) {
                    listener.onAction(new ActionResult("Knock"));
                }
            }
        }
         */
    }
}
