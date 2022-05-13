package com.hcifuture.contextactionlibrary.sensor.collector;

import java.util.HashMap;

public class CollectorStatusHolder {
    public enum CollectorStatus {
        READY,
        NOT_EXIST,
        UNKNOWN,
    }

    private final HashMap<Integer, CollectorStatus> statusMap;

    private static volatile CollectorStatusHolder instance = null;
    private CollectorStatusHolder() {
        statusMap = new HashMap<>();
    }

    public HashMap<Integer, CollectorStatus> getAllStatus() {
        return statusMap;
    }

    public CollectorStatus getStatus(Integer id) {
        if (statusMap.containsKey(id)) {
            return statusMap.get(id);
        }
        return CollectorStatus.UNKNOWN;
    }

    public void setStatus(Integer id, CollectorStatus status) {
        statusMap.put(id, status);
    }

    public void setStatus(Integer id, boolean status) {
        if (status) {
            statusMap.put(id, CollectorStatus.READY);
        } else {
            statusMap.put(id, CollectorStatus.NOT_EXIST);
        }
    }

    public static CollectorStatusHolder getInstance() {
        if (instance == null) {
            synchronized (CollectorStatusHolder.class) {
                if (instance == null) {
                    instance = new CollectorStatusHolder();
                }
            }
        }
        return instance;
    }

}
