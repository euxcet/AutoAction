package com.example.contextactionlibrary.collect.data;

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
            if (data.get(i).getName().equals(single.getName()) && data.get(i).getLinked() == single.getLinked()) {
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
