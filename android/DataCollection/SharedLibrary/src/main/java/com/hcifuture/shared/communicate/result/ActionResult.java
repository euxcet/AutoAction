package com.hcifuture.shared.communicate.result;

public class ActionResult extends Result {
    public ActionResult(String action) {
        super(action);
    }

    public ActionResult(String action, String reason) {
        super(action, reason);
    }

    public String getAction() {
        return super.getKey();
    }

    public void setAction(String action) {
        super.setKey(action);
    }
}
