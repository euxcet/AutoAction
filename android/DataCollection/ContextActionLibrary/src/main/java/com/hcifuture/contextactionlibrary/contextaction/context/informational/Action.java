package com.hcifuture.contextactionlibrary.contextaction.context.informational;

import java.util.HashSet;
import java.util.Set;
public class Action {
    String text;
    String type;

    public Action()
    {}

    public Action(String text,String type)
    {
        this.text = text;
        this.type = type;
    }

    @Override
    public String toString() {
        return "Action{" +
                "text='" + text + '\'' +
                ",type='" + type + '\'' +
                '}';
    }

    public String getType() {
        return type;
    }

    public void setText(String text) {
        this.text = text;
    }
}

