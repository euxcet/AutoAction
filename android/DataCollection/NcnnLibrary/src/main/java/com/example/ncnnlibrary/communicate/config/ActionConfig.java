package com.example.ncnnlibrary.communicate.config;

import com.example.ncnnlibrary.communicate.BuiltInActionEnum;
import com.example.ncnnlibrary.communicate.SensorType;

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
