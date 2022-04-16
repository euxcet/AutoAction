package com.hcifuture.contextactionlibrary.sensor.data;

public abstract class Data {
    public enum DataType {
        BluetoothData,
        SingleBluetoothData,
        IMUData,
        SingleIMUData,
        NonIMUData,
        LocationData,
        WeatherData,
        WifiData,
        SingleWifiData,
        LogData
    }

    public abstract DataType dataType();
}
