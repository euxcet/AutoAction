package com.hcifuture.contextactionlibrary.collect.data;

import java.util.ArrayList;
import java.util.List;

public class BluetoothData extends Data {
    private List<SingleBluetoothData> data;

    public BluetoothData() {
        data = new ArrayList<>();
    }

    public void setData(List<SingleBluetoothData> data) {
        this.data = data;
    }

    public List<SingleBluetoothData> getData() {
        return data;
    }

    public void clear() {
        this.data.clear();
    }

    public void insert(SingleBluetoothData single) {
        for (int i = 0; i < data.size(); i++) {
            SingleBluetoothData old = data.get(i);
            if (old.getAddress().equals(single.getAddress()) && old.getLinked() == single.getLinked()) {
                if (single.getScanResult().equals("") && !old.getScanResult().equals("")) {
                    single.setScanResult(old.getScanResult());
                }
                if (single.getIntentExtra().equals("") && !old.getIntentExtra().equals("")) {
                    single.setIntentExtra(old.getIntentExtra());
                }
                data.set(i, single);
                return;
            }
        }
        data.add(single);
    }

    public BluetoothData deepClone() {
        BluetoothData data = new BluetoothData();
        getData().forEach(data::insert);
        return data;
    }
}
