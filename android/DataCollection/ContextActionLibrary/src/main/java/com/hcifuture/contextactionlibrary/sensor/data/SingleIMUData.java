package com.hcifuture.contextactionlibrary.sensor.data;

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SingleIMUData extends Data {
    private List<Float> values;
    private String name;
    private int type;
    private long timestamp;

    public SingleIMUData(List<Float> values, String name, int type, long timestamp) {
        this.values = values;
        this.name = name;
        this.type = type;
        this.timestamp = timestamp;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public List<Float> getValues() {
        return values;
    }

    public void setValues(List<Float> values) {
        this.values = values;
    }

    public SingleIMUData deepClone() {
        List<Float> v = Arrays.asList(values.get(0), values.get(1), values.get(2));
        return new SingleIMUData(v, name, type, timestamp);
    }

    @Override
    public DataType dataType() {
        return DataType.SingleIMUData;
    }
}
