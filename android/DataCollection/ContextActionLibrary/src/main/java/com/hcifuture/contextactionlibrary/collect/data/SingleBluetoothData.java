package com.hcifuture.contextactionlibrary.collect.data;

public class SingleBluetoothData extends Data {
    private String name;
    private String address;
    private int bondState;
    private int type;
    private int deviceClass;
    private int majorDeviceClass;
    private short rssi;
    private boolean linked;
    private String scanResult;
    private String intentExtra;

    public SingleBluetoothData() {
        this.name = "";
        this.address = "";
        this.bondState = 0;
        this.type = 0;
        this.deviceClass = 0;
        this.majorDeviceClass = 0;
        this.rssi = 0;
        this.linked = false;
        this.scanResult = "";
        this.intentExtra = "";
    }

    public SingleBluetoothData(String name, String address,
                               int bondState, int type,
                               int deviceClass, int majorDeviceClass,
                               short rssi, boolean linked,
                               String scanResult, String intentExtra) {
        setName(name);
        setAddress(address);
        this.bondState = bondState;
        this.type = type;
        this.deviceClass = deviceClass;
        this.majorDeviceClass = majorDeviceClass;
        this.rssi = rssi;
        this.linked = linked;
        setScanResult(scanResult);
        setIntentExtra(intentExtra);
    }

    public int getBondState() {
        return bondState;
    }

    public int getType() {
        return type;
    }

    public String getAddress() {
        return address;
    }

    public String getName() {
        return name;
    }

    public int getDeviceClass() {
        return deviceClass;
    }

    public int getMajorDeviceClass() {
        return majorDeviceClass;
    }

    public boolean getLinked() {
        return linked;
    }

    public short getRssi() {
        return rssi;
    }

    public String getScanResult() {
        return scanResult;
    }

    public String getIntentExtra() {
        return intentExtra;
    }

    public void setAddress(String address) {
        if (address == null) {
            this.address = "";
        } else {
            this.address = address;
        }
    }

    public void setBondState(int bondState) {
        this.bondState = bondState;
    }

    public void setName(String name) {
        if (name == null) {
            this.name = "";
        } else {
            this.name = name;
        }
    }

    public void setType(int type) {
        this.type = type;
    }

    public void setDeviceClass(int deviceClass) {
        this.deviceClass = deviceClass;
    }

    public void setMajorDeviceClass(int majorDeviceClass) {
        this.majorDeviceClass = majorDeviceClass;
    }

    public void setRssi(short rssi) {
        this.rssi = rssi;
    }

    public void setLinked(boolean linked) {
        this.linked = linked;
    }

    public void setScanResult(String scanResult) {
        if (scanResult == null) {
            this.scanResult = "";
        } else {
            this.scanResult = scanResult;
        }
    }

    public void setIntentExtra(String intentExtra) {
        if (intentExtra == null) {
            this.intentExtra = "";
        } else {
            this.intentExtra = intentExtra;
        }
    }
}
