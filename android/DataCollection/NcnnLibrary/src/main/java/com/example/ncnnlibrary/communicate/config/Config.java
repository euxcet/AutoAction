package com.example.ncnnlibrary.communicate.config;

import com.example.ncnnlibrary.communicate.SensorType;

import java.util.HashMap;
import java.util.List;

public class Config {
    private final HashMap<String, Integer> integerConfig = new HashMap<>();
    private final HashMap<String, Long> longConfig = new HashMap<>();
    private final HashMap<String, Float> floatConfig = new HashMap<>();

    private List<SensorType> sensorType;

    public void putValue(String key, Integer value) {
        integerConfig.put(key, value);
    }

    public void putValue(String key, Long value) {
        longConfig.put(key, value);
    }

    public void putValue(String key, Float value) {
        floatConfig.put(key, value);
    }

    public Number getValue(String key) {
        if (integerConfig.containsKey(key)) {
            return integerConfig.get(key);
        }
        if (longConfig.containsKey(key)) {
            return longConfig.get(key);
        }
        if (floatConfig.containsKey(key)) {
            return floatConfig.get(key);
        }
        return null;
    }

    public List<SensorType> getSensorType() {
        return sensorType;
    }

    public void setSensorType(List<SensorType> sensorType) {
        this.sensorType = sensorType;
    }
}
