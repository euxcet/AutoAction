package com.hcifuture.shared.communicate.status;

import java.util.HashMap;

public class Heartbeat {
    private long timestamp;
    private int futureCount;
    private int aliveFutureCount;
    private HashMap<String, Long> lastSensorGetTimestamp;
    private HashMap<String, Long> lastCollectorAliveTimestamp;
    private HashMap<String, Long> lastActionAliveTimestamp;
    private HashMap<String, Long> lastContextAliveTimestamp;
    private HashMap<String, Long> lastActionTriggerTimestamp;
    private HashMap<String, Long> lastContextTriggerTimestamp;

    public Heartbeat(long timestamp, int futureCount, int aliveFutureCount,
                     HashMap<String, Long> lastSensorGetTimestamp,
                     HashMap<String, Long> lastCollectorAliveTimestamp,
                     HashMap<String, Long> lastActionAliveTimestamp,
                     HashMap<String, Long> lastContextAliveTimestamp,
                     HashMap<String, Long> lastActionTriggerTimestamp,
                     HashMap<String, Long> lastContextTriggerTimestamp) {
        this.timestamp = timestamp;
        this.futureCount = futureCount;
        this.aliveFutureCount = aliveFutureCount;
        this.lastSensorGetTimestamp = lastSensorGetTimestamp;
        this.lastCollectorAliveTimestamp = lastCollectorAliveTimestamp;
        this.lastActionAliveTimestamp = lastActionAliveTimestamp;
        this.lastContextAliveTimestamp = lastContextAliveTimestamp;
        this.lastActionTriggerTimestamp = lastActionTriggerTimestamp;
        this.lastContextTriggerTimestamp = lastContextTriggerTimestamp;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public HashMap<String, Long> getLastActionAliveTimestamp() {
        return lastActionAliveTimestamp;
    }

    public HashMap<String, Long> getLastActionTriggerTimestamp() {
        return lastActionTriggerTimestamp;
    }

    public HashMap<String, Long> getLastCollectorAliveTimestamp() {
        return lastCollectorAliveTimestamp;
    }

    public HashMap<String, Long> getLastContextAliveTimestamp() {
        return lastContextAliveTimestamp;
    }

    public HashMap<String, Long> getLastContextTriggerTimestamp() {
        return lastContextTriggerTimestamp;
    }

    public HashMap<String, Long> getLastSensorGetTimestamp() {
        return lastSensorGetTimestamp;
    }

    public int getAliveFutureCount() {
        return aliveFutureCount;
    }

    public int getFutureCount() {
        return futureCount;
    }

    public void setAliveFutureCount(int aliveFutureCount) {
        this.aliveFutureCount = aliveFutureCount;
    }

    public void setFutureCount(int futureCount) {
        this.futureCount = futureCount;
    }

    public void setLastActionAliveTimestamp(HashMap<String, Long> lastActionAliveTimestamp) {
        this.lastActionAliveTimestamp = lastActionAliveTimestamp;
    }

    public void setLastActionTriggerTimestamp(HashMap<String, Long> lastActionTriggerTimestamp) {
        this.lastActionTriggerTimestamp = lastActionTriggerTimestamp;
    }

    public void setLastCollectorAliveTimestamp(HashMap<String, Long> lastCollectorAliveTimestamp) {
        this.lastCollectorAliveTimestamp = lastCollectorAliveTimestamp;
    }

    public void setLastContextAliveTimestamp(HashMap<String, Long> lastContextAliveTimestamp) {
        this.lastContextAliveTimestamp = lastContextAliveTimestamp;
    }

    public void setLastContextTriggerTimestamp(HashMap<String, Long> lastContextTriggerTimestamp) {
        this.lastContextTriggerTimestamp = lastContextTriggerTimestamp;
    }

    public void setLastSensorGetTimestamp(HashMap<String, Long> lastSensorGetTimestamp) {
        this.lastSensorGetTimestamp = lastSensorGetTimestamp;
    }
}
