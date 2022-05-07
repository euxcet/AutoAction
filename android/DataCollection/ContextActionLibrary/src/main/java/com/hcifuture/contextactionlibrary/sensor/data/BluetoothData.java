package com.hcifuture.contextactionlibrary.sensor.data;

import android.os.Build;

import androidx.annotation.RequiresApi;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BluetoothData extends Data {
    private final List<SingleBluetoothData> devices;
    private String address;
    private int leMaximumAdvertisingDataLength;
    private String name;
    private int profileConnectionState_A2DP;
    private int profileConnectionState_HEADSET;
    private int scanMode;
    private int state;
    private boolean isDiscovering;
    private boolean isLe2MPhySupported;
    private boolean isLeCodedPhySupported;
    private boolean isLeExtendedAdvertisingSupported;
    private boolean isLePeriodicAdvertisingSupported;
    private boolean isMultipleAdvertisementSupported;
    private boolean isOffloadedFilteringSupported;
    private boolean isOffloadedScanBatchingSupported;

    public BluetoothData() {
        devices = Collections.synchronizedList(new ArrayList<>());
    }

    public String getAddress() {
        return address;
    }

    public int getLeMaximumAdvertisingDataLength() {
        return leMaximumAdvertisingDataLength;
    }

    public String getName() {
        return name;
    }

    public int getProfileConnectionState_A2DP() {
        return profileConnectionState_A2DP;
    }

    public int getProfileConnectionState_HEADSET() {
        return profileConnectionState_HEADSET;
    }

    public int getScanMode() {
        return scanMode;
    }

    public int getState() {
        return state;
    }

    public boolean isDiscovering() {
        return isDiscovering;
    }

    public boolean isLe2MPhySupported() {
        return isLe2MPhySupported;
    }

    public boolean isLeCodedPhySupported() {
        return isLeCodedPhySupported;
    }

    public boolean isLeExtendedAdvertisingSupported() {
        return isLeExtendedAdvertisingSupported;
    }

    public boolean isLePeriodicAdvertisingSupported() {
        return isLePeriodicAdvertisingSupported;
    }

    public boolean isMultipleAdvertisementSupported() {
        return isMultipleAdvertisementSupported;
    }

    public boolean isOffloadedFilteringSupported() {
        return isOffloadedFilteringSupported;
    }

    public boolean isOffloadedScanBatchingSupported() {
        return isOffloadedScanBatchingSupported;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public void setDiscovering(boolean discovering) {
        isDiscovering = discovering;
    }

    public void setLeMaximumAdvertisingDataLength(int leMaximumAdvertisingDataLength) {
        this.leMaximumAdvertisingDataLength = leMaximumAdvertisingDataLength;
    }

    public void setLe2MPhySupported(boolean le2MPhySupported) {
        isLe2MPhySupported = le2MPhySupported;
    }

    public void setLeCodedPhySupported(boolean leCodedPhySupported) {
        isLeCodedPhySupported = leCodedPhySupported;
    }

    public void setLeExtendedAdvertisingSupported(boolean leExtendedAdvertisingSupported) {
        isLeExtendedAdvertisingSupported = leExtendedAdvertisingSupported;
    }

    public void setLePeriodicAdvertisingSupported(boolean lePeriodicAdvertisingSupported) {
        isLePeriodicAdvertisingSupported = lePeriodicAdvertisingSupported;
    }

    public void setMultipleAdvertisementSupported(boolean multipleAdvertisementSupported) {
        isMultipleAdvertisementSupported = multipleAdvertisementSupported;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setOffloadedFilteringSupported(boolean offloadedFilteringSupported) {
        isOffloadedFilteringSupported = offloadedFilteringSupported;
    }

    public void setOffloadedScanBatchingSupported(boolean offloadedScanBatchingSupported) {
        isOffloadedScanBatchingSupported = offloadedScanBatchingSupported;
    }

    public void setProfileConnectionState_A2DP(int profileConnectionState_A2DP) {
        this.profileConnectionState_A2DP = profileConnectionState_A2DP;
    }

    public void setProfileConnectionState_HEADSET(int profileConnectionState_HEADSET) {
        this.profileConnectionState_HEADSET = profileConnectionState_HEADSET;
    }

    public void setScanMode(int scanMode) {
        this.scanMode = scanMode;
    }

    public void setState(int state) {
        this.state = state;
    }

    public void clear() {
        synchronized (devices) {
            devices.clear();
        }
    }

    public void insert(SingleBluetoothData single) {
        synchronized (devices) {
            for (int i = 0; i < devices.size(); i++) {
                SingleBluetoothData old = devices.get(i);
                if (old.getDevice().getAddress().equals(single.getDevice().getAddress()) && old.getLinked() == single.getLinked()) {
                    if (single.getIntentExtra() == null && old.getIntentExtra() != null) {
                        single.setIntentExtra(old.getIntentExtra());
                    }
                    if (single.getScanResult() == null && old.getScanResult() != null) {
                        single.setScanResult(old.getScanResult());
                    }
                    devices.set(i, single);
                    return;
                }
            }
            devices.add(single);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public BluetoothData deepClone() {
        BluetoothData bluetoothData = new BluetoothData();
        synchronized (devices) {
            devices.forEach(bluetoothData::insert);
        }
        bluetoothData.setAddress(getAddress());
        bluetoothData.setLeMaximumAdvertisingDataLength(getLeMaximumAdvertisingDataLength());
        bluetoothData.setName(getName());
        bluetoothData.setProfileConnectionState_A2DP(getProfileConnectionState_A2DP());
        bluetoothData.setProfileConnectionState_HEADSET(getProfileConnectionState_HEADSET());
        bluetoothData.setScanMode(getScanMode());
        bluetoothData.setState(getState());
        bluetoothData.setDiscovering(isDiscovering());
        bluetoothData.setLe2MPhySupported(isLe2MPhySupported());
        bluetoothData.setLeCodedPhySupported(isLeCodedPhySupported());
        bluetoothData.setLeExtendedAdvertisingSupported(isLeExtendedAdvertisingSupported());
        bluetoothData.setLePeriodicAdvertisingSupported(isLePeriodicAdvertisingSupported());
        bluetoothData.setMultipleAdvertisementSupported(isMultipleAdvertisementSupported());
        bluetoothData.setOffloadedFilteringSupported(isOffloadedFilteringSupported());
        bluetoothData.setOffloadedScanBatchingSupported(isOffloadedScanBatchingSupported());
        return bluetoothData;
    }

    @Override
    public DataType dataType() {
        return DataType.BluetoothData;
    }
}
