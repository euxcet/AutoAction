package com.example.contextactionlibrary.contextaction.action;

import com.example.contextactionlibrary.contextaction.action.BuiltInActionEnum;

import java.util.HashMap;

public class ActionConfig {
    private BuiltInActionEnum actionEnum;
    private final HashMap<String, Integer> integerConfig = new HashMap<>();
    private final HashMap<String, Long> longConfig = new HashMap<>();
    private final HashMap<String, Float> floatConfig = new HashMap<>();

    public void setActionEnum(BuiltInActionEnum actionEnum) {
        this.actionEnum = actionEnum;
    }

    public BuiltInActionEnum getActionEnum() {
        return actionEnum;
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
