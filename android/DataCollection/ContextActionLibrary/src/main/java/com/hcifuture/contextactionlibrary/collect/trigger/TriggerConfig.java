package com.hcifuture.contextactionlibrary.collect.trigger;

public class TriggerConfig {
    private int audioLength = 5000;
    private int bluetoothScanTime = 10000;
    private int wifiScanTime = 10000;
    public TriggerConfig() {
    }

    public int getAudioLength() {
        return audioLength;
    }

    public int getBluetoothScanTime() {
        return bluetoothScanTime;
    }

    public int getWifiScanTime() {
        return wifiScanTime;
    }

    public TriggerConfig setAudioLength(int audioLength) {
        this.audioLength = audioLength;
        return this;
    }

    public TriggerConfig setBluetoothScanTime(int bluetoothScanTime) {
        this.bluetoothScanTime = bluetoothScanTime;
        return this;
    }

    public TriggerConfig setWifiScanTime(int wifiScanTime) {
        this.wifiScanTime = wifiScanTime;
        return this;
    }
}
