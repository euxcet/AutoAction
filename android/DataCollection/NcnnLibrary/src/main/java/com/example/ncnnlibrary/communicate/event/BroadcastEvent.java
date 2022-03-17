package com.example.ncnnlibrary.communicate.event;

public class BroadcastEvent {
    private String action;
    private String tag;
    private String type;
    private int intValue;

    public BroadcastEvent(String action, String tag, String type) {
        this(action, tag, type, 0);
    }

    public BroadcastEvent(String action, String tag, String type, int intValue) {
        this.action = action;
        this.tag = tag;
        this.type = type;
        this.intValue = intValue;
    }

    public String getAction() {
        return action;
    }

    public String getTag() {
        return tag;
    }

    public int getIntValue() {
        return intValue;
    }

    public String getType() {
        return type;
    }
}
