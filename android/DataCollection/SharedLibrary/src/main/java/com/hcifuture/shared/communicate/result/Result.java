package com.hcifuture.shared.communicate.result;

import android.os.Bundle;

public class Result {
    private String key;
    private String reason;
    private long timestamp;
    private Bundle extras;

    public Result(String key) {
        this(key, null);
    }

    public Result(String key, String reason) {
        setKey(key);
        setReason(reason);
    }

    public void setKey(String key) {
        if (key == null) {
            this.key = "NULL";
        } else {
            this.key = key;
        }
    }

    public void setReason(String reason) {
        if (reason == null) {
            this.reason = "NULL";
        } else {
            this.reason = reason;
        }
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public void setExtras(Bundle extras) {
        this.extras = (extras == null)? new Bundle() : extras;
    }

    public String getKey() {
        return key;
    }

    public String getReason() {
        return reason;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public Bundle getExtras() {
        return extras;
    }
}
