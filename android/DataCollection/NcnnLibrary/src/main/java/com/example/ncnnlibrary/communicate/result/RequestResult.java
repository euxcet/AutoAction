package com.example.ncnnlibrary.communicate.result;

import java.util.HashMap;

public class RequestResult {
    private final HashMap<String, Boolean> booleanConfig = new HashMap<>();
    private final HashMap<String, Integer> integerConfig = new HashMap<>();
    private final HashMap<String, Long> longConfig = new HashMap<>();
    private final HashMap<String, Float> floatConfig = new HashMap<>();

    public void putValue(String key, Boolean value) {
        booleanConfig.put(key, value);
    }

    public void putValue(String key, Integer value) {
        integerConfig.put(key, value);
    }

    public void putValue(String key, Long value) {
        longConfig.put(key, value);
    }

    public void putValue(String key, Float value) {
        floatConfig.put(key, value);
    }

    public Boolean getBoolean(String key) {
        if (booleanConfig.containsKey(key)) {
            return booleanConfig.get(key);
        }
        return null;
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
}
