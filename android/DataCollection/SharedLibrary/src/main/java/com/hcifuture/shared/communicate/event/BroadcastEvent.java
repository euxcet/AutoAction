package com.hcifuture.shared.communicate.event;

import android.os.Bundle;

public class BroadcastEvent {
    private String action;
    private String tag;
    private String type;
    private Bundle extras;

    public BroadcastEvent(String action, String tag, String type) {
        this(action, tag, type, null);
    }

    public BroadcastEvent(String action, Bundle extras) {
        this(action, "", "", extras);
    }

    public BroadcastEvent(String action, String tag, String type, Bundle extras) {
        setAction(action);
        setTag(tag);
        setType(type);
        setExtras(extras);
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
        return extras;
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
