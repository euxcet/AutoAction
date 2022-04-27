package com.hcifuture.contextactionlibrary.contextaction.event;

import android.os.Bundle;

public class BroadcastEvent {
    private long timestamp;
    private String action;
    private String tag;
    private String type;
    private Bundle extras;

    public BroadcastEvent(long timestamp, String action, String tag, String type) {
        setTimestamp(timestamp);
        setAction(action);
        setTag(tag);
        setType(type);
    }

    public long getTimestamp() {
        return timestamp;
    }

    public String getAction() {
        return action;
    }

    public String getTag() {
        return tag;
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

    public void setTag(String tag) {
        this.tag = (tag == null)? "" : tag;
    }

    public void setType(String type) {
        this.type = (type == null)? "" : type;
    }

    public void setExtras(Bundle extras) {
        this.extras = (extras == null)? new Bundle() : extras;
    }
}
