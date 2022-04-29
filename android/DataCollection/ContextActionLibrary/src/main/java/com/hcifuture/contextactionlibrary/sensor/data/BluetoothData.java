package com.hcifuture.contextactionlibrary.sensor.data;

import android.os.Build;

import androidx.annotation.RequiresApi;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BluetoothData extends Data {
    private final List<SingleBluetoothData> data;

    public BluetoothData() {
        data = Collections.synchronizedList(new ArrayList<>());
    }

    public void clear() {
        synchronized (data) {
            data.clear();
        }
    }

    public void insert(SingleBluetoothData single) {
        synchronized (data) {
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
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public BluetoothData deepClone() {
        synchronized (data) {
            BluetoothData bluetoothData = new BluetoothData();
            data.forEach(bluetoothData::insert);
            return bluetoothData;
        }
    }

    @Override
    public DataType dataType() {
        return DataType.BluetoothData;
    }
}
