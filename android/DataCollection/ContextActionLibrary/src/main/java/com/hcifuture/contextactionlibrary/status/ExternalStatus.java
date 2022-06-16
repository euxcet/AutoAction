package com.hcifuture.contextactionlibrary.status;

import java.util.HashMap;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class ExternalStatus {
    private static volatile ExternalStatus instance = null;
    private final HashMap<String, Integer> integerStatusMap;
    private final HashMap<String, Long> longStatusMap;
    private final HashMap<String, Float> floatStatusMap;
    private final HashMap<String, Boolean> booleanStatusMap;
    private final HashMap<String, String> stringStatusMap;

    private ExternalStatus() {
        integerStatusMap = new HashMap<>();
        longStatusMap = new HashMap<>();
        booleanStatusMap = new HashMap<>();
        floatStatusMap = new HashMap<>();
        stringStatusMap = new HashMap<>();
    }

    public synchronized void setBooleanStatus(String key, Boolean value) {
        booleanStatusMap.put(key, value);
    }

    public synchronized void setIntegerStatus(String key, Integer value) {
        integerStatusMap.put(key, value);
    }

    public synchronized void setLongStatus(String key, Long value) {
        longStatusMap.put(key, value);
    }

    public synchronized void setFloatStatus(String key, Float value) {
        floatStatusMap.put(key, value);
    }

    public synchronized void setStringStatus(String key, String value) {
        stringStatusMap.put(key, value);
    }

    public synchronized Boolean getBooleanStatus(String key) {
        return booleanStatusMap.get(key);
    }

    public synchronized Long getLongStatus(String key) {
        return longStatusMap.get(key);
    }

    public synchronized Integer getIntegerStatus(String key) {
        return integerStatusMap.get(key);
    }

    public synchronized Float getFloatStatus(String key) {
        return floatStatusMap.get(key);
    }

    public synchronized String getStringStatus(String key) {
        return stringStatusMap.get(key);
    }

    public static ExternalStatus getInstance() {
        if (instance == null) {
            synchronized (Heart.class) {
                if (instance == null) {
                    instance = new ExternalStatus();
                }
            }
        }
        return instance;
    }
}
