package com.hcifuture.contextactionlibrary.sensor.data;

import android.os.Build;

import androidx.annotation.RequiresApi;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class WifiData extends Data {
    private final List<SingleWifiData> data;

    public WifiData() {
        data = Collections.synchronizedList(new ArrayList<>());
    }

    public void clear() {
        synchronized (data) {
            data.clear();
        }
    }

    public void insert(SingleWifiData single) {
        synchronized (data) {
            for (int i = 0; i < data.size(); i++) {
                if (data.get(i).getBssid().equals(single.getBssid()) && data.get(i).getConnected() == single.getConnected()) {
                    data.set(i, single);
                    return;
                }
            }
            data.add(single);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public WifiData deepClone() {
        synchronized (data) {
            WifiData wifiData = new WifiData();
            data.forEach(wifiData::insert);
            return wifiData;
        }
    }

    @Override
    public DataType dataType() {
        return DataType.WifiData;
    }
}
