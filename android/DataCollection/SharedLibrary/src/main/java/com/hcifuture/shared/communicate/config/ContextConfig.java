package com.hcifuture.shared.communicate.config;

import com.hcifuture.shared.communicate.SensorType;

import java.util.List;

public class ContextConfig extends Config {
    private String context;
    private List<SensorType> sensorType;

    public String getContext() {
        return context;
    }

    public void setContext(String context) {
        this.context = context;
    }

    public List<SensorType> getSensorType() {
        return sensorType;
    }

    public void setSensorType(List<SensorType> sensorType) {
        this.sensorType = sensorType;
    }
}
