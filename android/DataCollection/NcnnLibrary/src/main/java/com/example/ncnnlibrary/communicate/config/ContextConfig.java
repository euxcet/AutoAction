package com.example.ncnnlibrary.communicate.config;

import com.example.ncnnlibrary.communicate.BuiltInContextEnum;
import com.example.ncnnlibrary.communicate.SensorType;

import java.util.List;

public class ContextConfig extends Config {
    private BuiltInContextEnum context;
    private List<SensorType> sensorType;

    public BuiltInContextEnum getContext() {
        return context;
    }

    public void setContext(BuiltInContextEnum context) {
        this.context = context;
    }

    public List<SensorType> getSensorType() {
        return sensorType;
    }

    public void setSensorType(List<SensorType> sensorType) {
        this.sensorType = sensorType;
    }
}
