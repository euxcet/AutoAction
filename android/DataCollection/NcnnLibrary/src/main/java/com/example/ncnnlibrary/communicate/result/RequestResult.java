package com.example.ncnnlibrary.communicate.result;

import java.util.HashMap;

public class RequestResult {
    private final HashMap<String, Boolean> booleanResult = new HashMap<>();
    private final HashMap<String, Integer> integerResult = new HashMap<>();
    private final HashMap<String, Long> longResult = new HashMap<>();
    private final HashMap<String, Float> floatResult = new HashMap<>();
    private final HashMap<String, Object> objectResult = new HashMap<>();

    public void putValue(String key, Boolean value) {
        booleanResult.put(key, value);
    }

    public void putValue(String key, Integer value) {
        integerResult.put(key, value);
    }

    public void putValue(String key, Long value) {
        longResult.put(key, value);
    }

    public void putValue(String key, Float value) {
        floatResult.put(key, value);
    }

    public void putObject(String key, Object value) {
        objectResult.put(key, value);
    }

    public Boolean getBoolean(String key) {
        if (booleanResult.containsKey(key)) {
            return booleanResult.get(key);
        }
        return null;
    }

    public Number getValue(String key) {
        if (integerResult.containsKey(key)) {
            return integerResult.get(key);
        }
        if (longResult.containsKey(key)) {
            return longResult.get(key);
        }
        if (floatResult.containsKey(key)) {
            return floatResult.get(key);
        }
        return null;
    }

    public Object getObject(String key) {
        if (objectResult.containsKey(key)) {
            return objectResult.get(key);
        }
        return null;
    }
}
