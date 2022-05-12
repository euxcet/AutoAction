package com.hcifuture.contextactionlibrary.sensor.distributor;

import android.util.Log;

import com.hcifuture.contextactionlibrary.contextaction.action.BaseAction;
import com.hcifuture.contextactionlibrary.contextaction.context.BaseContext;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorListener;
import com.hcifuture.contextactionlibrary.sensor.data.Data;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.shared.communicate.SensorType;

import java.util.List;

public class DataDistributor implements CollectorListener {
    private List<BaseAction> actions;
    private List<BaseContext> contexts;
    public DataDistributor(List<BaseAction> actions, List<BaseContext> contexts) {
        this.actions = actions;
        this.contexts = contexts;
    }

    @Override
    public void onSensorEvent(Data data) {
        for (BaseAction action: actions) {
            List<SensorType> types = action.getConfig().getSensorType();
            switch (data.dataType()) {
                case SingleIMUData:
                    if (types.contains(SensorType.IMU)) {
                        action.onIMUSensorEvent((SingleIMUData)data);
                    }
                    break;
                case NonIMUData:
                    if (types.contains(SensorType.PROXIMITY)) {
                        Log.e("proximity:","onNonIMUSensorChanged");
                        action.onNonIMUSensorEvent((NonIMUData)data);
                    }
                    break;
                default:
                    break;
            }
        }
        for (BaseContext context: contexts) {
            List<SensorType> types = context.getConfig().getSensorType();
            switch (data.dataType()) {
                case SingleIMUData:
                    if (types.contains(SensorType.IMU)) {
                        context.onIMUSensorEvent((SingleIMUData)data);
                    }
                    break;
                case NonIMUData:
                    if (types.contains(SensorType.PROXIMITY)) {
                        context.onNonIMUSensorEvent((NonIMUData)data);
                    }
                    break;
                default:
                    break;
            }
        }
    }
}
