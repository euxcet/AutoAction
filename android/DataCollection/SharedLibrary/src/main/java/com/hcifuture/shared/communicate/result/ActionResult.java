package com.hcifuture.shared.communicate.result;

public class ActionResult {
    private String action;
    private String reason;
    private String timestamp;

    public ActionResult(String action) {
        this.action = action;
    }

    public void setAction(String action) {
        this.action = action;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    public String getAction() {
        return action;
    }

    public String getReason() {
        return reason;
    }

    public String getTimestamp() {
        return timestamp;
    }
}
