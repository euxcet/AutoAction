package com.hcifuture.shared.communicate.config;

import com.hcifuture.shared.communicate.SensorType;

import java.util.List;

public class ActionConfig extends Config {
    private String action;
    private List<SensorType> sensorType;

    public String getAction() {
        return action;
    }

    public void setAction(String action) {
        this.action = action;
    }

    public List<SensorType> getSensorType() {
        return sensorType;
    }

    public void setSensorType(List<SensorType> sensorType) {
        this.sensorType = sensorType;
    }
}
