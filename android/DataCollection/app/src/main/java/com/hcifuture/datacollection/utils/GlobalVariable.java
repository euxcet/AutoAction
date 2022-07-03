package com.hcifuture.datacollection.utils;

import java.util.HashMap;

/**
 * Map strings to strings globally.
 */
public class GlobalVariable {
    private static GlobalVariable instance = new GlobalVariable();
    private HashMap<String, String> stringHashMap;

    private GlobalVariable() {
        stringHashMap = new HashMap<>();
    }

    public void putString(String key, String value) {
        stringHashMap.put(key, value);
    }

    public String getString(String key) {
        return stringHashMap.get(key);
    }

    public static GlobalVariable getInstance() {
        return instance;
    }
}
