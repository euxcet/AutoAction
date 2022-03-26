package com.hcifuture.contextactionlibrary.collect.data;

public class NonIMUData extends Data {
    private float environmentBrightness;
    private float airPressure;
    private int screenBrightness;
    private long environmentBrightnessTimestamp;
    private long airPressureTimestamp;
    private long screenBrightnessTimestamp;

    public NonIMUData() {
        environmentBrightness = 0;
        airPressure = 0;
        screenBrightness = 0;
        environmentBrightnessTimestamp = 0;
        airPressureTimestamp = 0;
        screenBrightnessTimestamp = 0;
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

    public void setAirPressure(float airPressure) {
        this.airPressure = airPressure;
    }

    public void setEnvironmentBrightness(float environmentBrightness) {
        this.environmentBrightness = environmentBrightness;
    }

    public void setScreenBrightness(int screenBrightness) {
        this.screenBrightness = screenBrightness;
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

    public void setAirPressureTimestamp(long airPressureTimestamp) {
        this.airPressureTimestamp = airPressureTimestamp;
    }

    public void setEnvironmentBrightnessTimestamp(long environmentBrightnessTimestamp) {
        this.environmentBrightnessTimestamp = environmentBrightnessTimestamp;
    }

    public void setScreenBrightnessTimestamp(long screenBrightnessTimestamp) {
        this.screenBrightnessTimestamp = screenBrightnessTimestamp;
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
}
