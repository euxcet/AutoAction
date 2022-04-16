package com.hcifuture.contextactionlibrary.sensor.collector;

import com.hcifuture.contextactionlibrary.sensor.data.Data;

public interface CollectorListener {
    void onSensorEvent(Data data);
}
