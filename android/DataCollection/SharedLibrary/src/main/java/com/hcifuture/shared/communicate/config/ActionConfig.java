package com.hcifuture.shared.communicate.config;

import com.hcifuture.shared.communicate.SensorType;

import java.util.List;

public class ActionConfig extends Config {
    private String action;
    private int priority; // 0 necessary  1 unnecessary
    private int imuSamplingFreq;
    private List<SensorType> sensorType;

    public int getPriority() {
        return priority;
    }

    public void setPriority(int priority) {
        this.priority = priority;
    }

    public int getImuSamplingFreq() {
        return imuSamplingFreq;
    }

    public void setImuSamplingFreq(int imuSamplingFreq) {
        this.imuSamplingFreq = imuSamplingFreq;
    }

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
