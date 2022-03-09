package com.example.ncnnlibrary.communicate;

import java.util.HashMap;

public class ActionConfig {
    private BuiltInActionEnum action;
    private final HashMap<String, Integer> integerConfig = new HashMap<>();
    private final HashMap<String, Long> longConfig = new HashMap<>();
    private final HashMap<String, Float> floatConfig = new HashMap<>();

    public BuiltInActionEnum getAction() {
        return action;
    }

    public void setAction(BuiltInActionEnum action) {
        this.action = action;
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
