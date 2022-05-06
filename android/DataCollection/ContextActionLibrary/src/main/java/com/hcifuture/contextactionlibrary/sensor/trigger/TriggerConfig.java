package com.hcifuture.contextactionlibrary.sensor.trigger;

public class TriggerConfig {
    private int imuHead = -1;
    private int imuTail = -1;
    private int audioLength = 5000;
    private String audioFilename = "";
    private int bluetoothScanTime = 10000;
    private int wifiScanTimeout = 10000;
    private int gpsRequestTime = 3000;

    public TriggerConfig() {
    }

    public int getGPSRequestTime() {
        return gpsRequestTime;
    }

    public TriggerConfig setGPSRequestTime(int gpsRequestTime) {
        this.gpsRequestTime = gpsRequestTime;
        return this;
    }

    public int getImuHead() {
        return imuHead;
    }

    public int getImuTail() {
        return imuTail;
    }

    public TriggerConfig setImuHead(int imuHead) {
        this.imuHead = imuHead;
        return this;
    }

    public TriggerConfig setImuTail(int imuTail) {
        this.imuTail = imuTail;
        return this;
    }

    public int getAudioLength() {
        return audioLength;
    }

    public int getBluetoothScanTime() {
        return bluetoothScanTime;
    }

    public int getWifiScanTimeout() {
        return wifiScanTimeout;
    }

    public String getAudioFilename() {
        return audioFilename;
    }

    public TriggerConfig setAudioLength(int audioLength) {
        this.audioLength = audioLength;
        return this;
    }

    public TriggerConfig setBluetoothScanTime(int bluetoothScanTime) {
        this.bluetoothScanTime = bluetoothScanTime;
        return this;
    }

    public TriggerConfig setWifiScanTimeout(int wifiScanTimeout) {
        this.wifiScanTimeout = wifiScanTimeout;
        return this;
    }

    public TriggerConfig setAudioFilename(String audioFilename) {
        this.audioFilename = audioFilename;
        return this;
    }
}
