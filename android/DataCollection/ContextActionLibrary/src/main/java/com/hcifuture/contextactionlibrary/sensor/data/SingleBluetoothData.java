package com.hcifuture.contextactionlibrary.sensor.data;

import android.bluetooth.BluetoothDevice;
import android.bluetooth.le.ScanResult;
import android.os.Bundle;

import java.util.Objects;

public class SingleBluetoothData extends Data {
    private boolean linked;
    private BluetoothDevice device;
    private ScanResult scanResult;
    private Bundle intentExtra;

    public SingleBluetoothData(BluetoothDevice device,
                               boolean linked,
                               ScanResult scanResult,
                               Bundle intentExtra) {
        setDevice(device);
        setLinked(linked);
        setScanResult(scanResult);
        setIntentExtra(intentExtra);
    }

    public BluetoothDevice getDevice() {
        return device;
    }

    public boolean getLinked() {
        return linked;
    }

    public ScanResult getScanResult() {
        return scanResult;
    }

    public Bundle getIntentExtra() {
        return intentExtra;
    }

    public void setDevice(BluetoothDevice device) {
        Objects.requireNonNull(device);
        this.device = device;
    }

    public void setLinked(boolean linked) {
        this.linked = linked;
    }

    public void setScanResult(ScanResult scanResult) {
        this.scanResult = scanResult;
    }

    public void setIntentExtra(Bundle intentExtra) {
        this.intentExtra = intentExtra;
    }

    @Override
    public DataType dataType() {
        return DataType.SingleBluetoothData;
    }
}
