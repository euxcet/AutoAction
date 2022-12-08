package com.hcifuture.datacollection.action;

public class ActionResult {
    private ActionWithObject action;
    private float minDistance;
    private long timestamp;

    public ActionResult(ActionWithObject action, float minDistance, long timestamp) {
        this.action = action;
        this.minDistance = minDistance;
        this.timestamp = timestamp;
    }

    public void setAction(ActionWithObject action) {
        this.action = action;
    }

    public ActionWithObject getAction() {
        return action;
    }

    public float getMinDistance() {
        return minDistance;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setMinDistance(float minDistance) {
        this.minDistance = minDistance;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
}
