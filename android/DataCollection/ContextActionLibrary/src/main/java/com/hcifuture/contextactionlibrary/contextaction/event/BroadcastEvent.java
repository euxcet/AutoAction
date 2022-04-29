package com.hcifuture.contextactionlibrary.contextaction.event;

import android.os.Bundle;

public class BroadcastEvent {
    private long timestamp;
    private String type;
    private String action;
    private Bundle extras;

    public BroadcastEvent(long timestamp, String type, String action) {
        setTimestamp(timestamp);
        setType(type);
        setAction(action);
    }

    public long getTimestamp() {
        return timestamp;
    }

    public String getAction() {
        return action;
    }

    public String getType() {
        return type;
    }

    public Bundle getExtras() {
        if (extras == null) {
            extras = new Bundle();
        }
        return extras;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public void setAction(String action) {
        this.action = (action == null)? "" : action;
    }

    public void setType(String type) {
        this.type = (type == null)? "" : type;
    }

    public void setExtras(Bundle extras) {
        this.extras = (extras == null)? new Bundle() : extras;
    }
}
