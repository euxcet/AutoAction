package com.hcifuture.shared.communicate.result;

public class ContextResult {
    private String context;
    private long timestamp;

    public ContextResult(String context) {
        this.context = context;
    }

    public void setContext(String context) {
        this.context = context;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public String getContext() {
        return context;
    }

    public long getTimestamp() {
        return timestamp;
    }
}
