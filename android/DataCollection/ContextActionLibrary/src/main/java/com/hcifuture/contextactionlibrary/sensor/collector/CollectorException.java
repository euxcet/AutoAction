package com.hcifuture.contextactionlibrary.sensor.collector;

import androidx.annotation.NonNull;

public class CollectorException extends Exception {

    private final int code;

    public CollectorException(int code, String msg) {
        super(msg);
        this.code = code;
    }

    public CollectorException(int code, Throwable cause) {
        super(cause);
        this.code = code;
    }

    public int getCode() {
        return code;
    }

    @NonNull
    @Override
    public String toString() {
        return "CollectorException(" + code + "): " + getLocalizedMessage();
    }
}
