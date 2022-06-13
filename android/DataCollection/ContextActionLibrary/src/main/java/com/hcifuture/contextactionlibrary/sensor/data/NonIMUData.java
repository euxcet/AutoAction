package com.hcifuture.contextactionlibrary.sensor.data;

public class NonIMUData extends Data {
    private int type;
    private float environmentBrightness;
    private float airPressure;
    private int screenBrightness;
    private float proximity;
    private float stepCounter;
    private long environmentBrightnessTimestamp;
    private long airPressureTimestamp;
    private long screenBrightnessTimestamp;
    private long proximityTimestamp;
    private long stepCounterTimestamp;

    public NonIMUData() {
        environmentBrightness = 0;
        airPressure = 0;
        screenBrightness = 0;
        proximity = 0;
        stepCounter = 0;
        environmentBrightnessTimestamp = 0;
        airPressureTimestamp = 0;
        screenBrightnessTimestamp = 0;
        proximityTimestamp = 0;
        stepCounterTimestamp = 0;
    }

    public long getTimestamp() {
        return Math.max(Math.max(Math.max(environmentBrightnessTimestamp, airPressureTimestamp),
                        Math.max(screenBrightnessTimestamp, proximityTimestamp)), stepCounterTimestamp);
    }

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }

    public float getAirPressure() {
        return airPressure;
    }

    public float getEnvironmentBrightness() {
        return environmentBrightness;
    }

    public int getScreenBrightness() {
        return screenBrightness;
    }

    public float getProximity() {
        return proximity;
    }

    public float getStepCounter() {
        return stepCounter;
    }

    public void setAirPressure(float airPressure) {
        this.airPressure = airPressure;
    }

    public void setEnvironmentBrightness(float environmentBrightness) {
        this.environmentBrightness = environmentBrightness;
    }

    public void setScreenBrightness(int screenBrightness) {
        this.screenBrightness = screenBrightness;
    }

    public void setProximity(float proximity) {
        this.proximity = proximity;
    }

    public void setStepCounter(float stepCounter) {
        this.stepCounter = stepCounter;
    }

    public long getAirPressureTimestamp() {
        return airPressureTimestamp;
    }

    public long getEnvironmentBrightnessTimestamp() {
        return environmentBrightnessTimestamp;
    }

    public long getScreenBrightnessTimestamp() {
        return screenBrightnessTimestamp;
    }

    public long getProximityTimestamp() {
        return proximityTimestamp;
    }

    public long getStepCounterTimestamp() {
        return stepCounterTimestamp;
    }

    public void setAirPressureTimestamp(long airPressureTimestamp) {
        this.airPressureTimestamp = airPressureTimestamp;
    }

    public void setEnvironmentBrightnessTimestamp(long environmentBrightnessTimestamp) {
        this.environmentBrightnessTimestamp = environmentBrightnessTimestamp;
    }

    public void setScreenBrightnessTimestamp(long screenBrightnessTimestamp) {
        this.screenBrightnessTimestamp = screenBrightnessTimestamp;
    }

    public void setProximityTimestamp(long proximityTimestamp) {
        this.proximityTimestamp = proximityTimestamp;
    }

    public void setStepCounterTimestamp(long stepCounterTimestamp) {
        this.stepCounterTimestamp = stepCounterTimestamp;
    }

    public NonIMUData deepClone() {
        NonIMUData data = new NonIMUData();
        data.airPressure = airPressure;
        data.airPressureTimestamp = airPressureTimestamp;
        data.environmentBrightness = environmentBrightness;
        data.environmentBrightnessTimestamp = environmentBrightnessTimestamp;
        data.screenBrightness = screenBrightness;
        data.screenBrightnessTimestamp = screenBrightnessTimestamp;
        return data;
    }

    @Override
    public DataType dataType() {
        return DataType.NonIMUData;
    }
}
