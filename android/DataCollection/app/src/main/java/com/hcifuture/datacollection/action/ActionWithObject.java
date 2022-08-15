package com.hcifuture.datacollection.action;

public class ActionWithObject {
    private String name;
    private ActionEnum action;
    private ObjectDescriptor object;

    public ActionWithObject(String name, ActionEnum action, ObjectDescriptor object) {
        this.name = name;
        this.action = action;
        this.object = object;
    }

    public ActionEnum getAction() {
        return action;
    }

    public ObjectDescriptor getObject() {
        return object;
    }

    public void setAction(ActionEnum action) {
        this.action = action;
    }

    public void setObject(ObjectDescriptor object) {
        this.object = object;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
