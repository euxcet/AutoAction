package com.hcifuture.contextactionlibrary.collect.trigger;

public class TriggerConfig {
    private int audioLength;
    private int bluetoothScanTime;
    private int wifiScanTime;
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
