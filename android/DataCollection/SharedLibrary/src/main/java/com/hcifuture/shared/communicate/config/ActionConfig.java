package com.hcifuture.shared.communicate.config;

import com.hcifuture.shared.communicate.BuiltInActionEnum;
import com.hcifuture.shared.communicate.SensorType;

import java.util.List;

public class ActionConfig extends Config {
    private BuiltInActionEnum action;
    private List<SensorType> sensorType;

    public BuiltInActionEnum getAction() {
        return action;
    }

    public void setAction(BuiltInActionEnum action) {
        this.action = action;
    }

    public List<SensorType> getSensorType() {
        return sensorType;
    }

    public void setSensorType(List<SensorType> sensorType) {
        this.sensorType = sensorType;
    }
}
