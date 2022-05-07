package com.hcifuture.contextactionlibrary.sensor.data;

import android.os.Build;

import androidx.annotation.RequiresApi;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class WifiData extends Data {
    private final List<SingleWifiData> aps;
    private int state;

    public WifiData() {
        aps = Collections.synchronizedList(new ArrayList<>());
    }

    public int getState() {
        return state;
    }

    public void setState(int state) {
        this.state = state;
    }

    public void clear() {
        synchronized (aps) {
            aps.clear();
        }
    }

    public void insert(SingleWifiData single) {
        synchronized (aps) {
            for (int i = 0; i < aps.size(); i++) {
                if (aps.get(i).getBssid().equals(single.getBssid()) && aps.get(i).getConnected() == single.getConnected()) {
                    aps.set(i, single);
                    return;
                }
            }
            aps.add(single);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public WifiData deepClone() {
        WifiData wifiData = new WifiData();
        synchronized (aps) {
            aps.forEach(wifiData::insert);
        }
        wifiData.setState(getState());
        return wifiData;
    }

    @Override
    public DataType dataType() {
        return DataType.WifiData;
    }
}
