package com.hcifuture.shared.communicate.result;

public class ContextResult extends Result {
    public ContextResult(String context) {
        super(context);
    }

    public ContextResult(String context, String reason) {
        super(context, reason);
    }

    public String getContext() {
        return super.getKey();
    }

    public void setContext(String context) {
        super.setKey(context);
    }
}
