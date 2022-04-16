package com.hcifuture.contextactionlibrary.sensor.data;

import android.os.Build;

import androidx.annotation.RequiresApi;

import java.util.ArrayList;
import java.util.List;

public class WifiData extends Data {
    private List<SingleWifiData> data;

    public WifiData() {
        data = new ArrayList<>();
    }

    public List<SingleWifiData> getData() {
        return data;
    }

    public void setData(List<SingleWifiData> data) {
        this.data = data;
    }

    public void clear() {
        data.clear();
    }

    public void insert(SingleWifiData single) {
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).getBssid().equals(single.getBssid()) && data.get(i).getConnected() == single.getConnected()) {
                data.set(i, single);
                return;
            }
        }
        data.add(single);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public WifiData deepClone() {
        WifiData data = new WifiData();
        getData().forEach(data::insert);
        return data;
    }

    @Override
    public DataType dataType() {
        return DataType.WifiData;
    }
}
