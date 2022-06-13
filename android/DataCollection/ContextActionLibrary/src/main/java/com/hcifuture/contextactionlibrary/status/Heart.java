package com.hcifuture.contextactionlibrary.status;

import android.os.Build;

import com.hcifuture.shared.communicate.status.Heartbeat;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Heart {
    private static volatile Heart instance = null;
    private HashMap<String, Long> lastSensorGetTimestamp;
    private HashMap<String, Long> lastCollectorAliveTimestamp;
    private HashMap<String, Long> lastActionAliveTimestamp;
    private HashMap<String, Long> lastContextAliveTimestamp;
    private HashMap<String, Long> lastActionTriggerTimestamp;
    private HashMap<String, Long> lastContextTriggerTimestamp;
    private int futureCount = -1;
    private int aliveFutureCount = -1;

    private Heart() {
        lastSensorGetTimestamp = new HashMap<>();
        lastCollectorAliveTimestamp = new HashMap<>();
        lastActionAliveTimestamp = new HashMap<>();
        lastContextAliveTimestamp = new HashMap<>();
        lastActionTriggerTimestamp = new HashMap<>();
        lastContextTriggerTimestamp = new HashMap<>();
    }

    HashMap<String, Long> deepClone(HashMap<String, Long> data) {
        HashMap<String, Long> result = new HashMap<>();
        for (Map.Entry<String, Long> entry: data.entrySet()) {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }

    public synchronized Heartbeat beat() {
        return new Heartbeat(System.currentTimeMillis(), this.futureCount, this.aliveFutureCount,
                deepClone(this.lastSensorGetTimestamp),
                deepClone(this.lastCollectorAliveTimestamp),
                deepClone(this.lastActionAliveTimestamp),
                deepClone(this.lastContextAliveTimestamp),
                deepClone(this.lastActionTriggerTimestamp),
                deepClone(this.lastContextTriggerTimestamp));
    }

    public synchronized void newSensorGetEvent(String sensor, long timestamp) {
        lastSensorGetTimestamp.put(sensor, timestamp);
    }

    public synchronized void newCollectorAliveEvent(String collector, long timestamp) {
        lastCollectorAliveTimestamp.put(collector, timestamp);
    }

    public synchronized void newActionAliveEvent(String action, long timestamp) {
        lastActionAliveTimestamp.put(action, timestamp);
    }

    public synchronized void newContextAliveEvent(String context, long timestamp) {
        lastContextAliveTimestamp.put(context, timestamp);
    }

    public synchronized void newActionTriggerEvent(String action, long timestamp) {
        lastActionTriggerTimestamp.put(action, timestamp);
    }

    public synchronized void newContextTriggerEvent(String context, long timestamp) {
        lastContextTriggerTimestamp.put(context, timestamp);
    }

    public synchronized void setFutureCount(int futureCount) {
        this.futureCount = futureCount;
    }

    public synchronized void setAliveFutureCount(int aliveFutureCount) {
        this.aliveFutureCount = aliveFutureCount;
    }

    public static Heart getInstance() {
        if (instance == null) {
            synchronized (Heart.class) {
                if (instance == null) {
                    instance = new Heart();
                }
            }
        }
        return instance;
    }
}
