package com.hcifuture.shared.communicate.result;

public class ContextResult {
    private String context;
    private String timestamp;

    public ContextResult(String context) {
        this.context = context;
    }

    public void setContext(String context) {
        this.context = context;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    public String getContext() {
        return context;
    }

    public String getTimestamp() {
        return timestamp;
    }
}
