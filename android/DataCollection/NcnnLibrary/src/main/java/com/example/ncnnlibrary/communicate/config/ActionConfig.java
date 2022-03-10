package com.example.ncnnlibrary.communicate.config;

import com.example.ncnnlibrary.communicate.BuiltInActionEnum;

public class ActionConfig extends Config {
    private BuiltInActionEnum action;

    public BuiltInActionEnum getAction() {
        return action;
    }

    public void setAction(BuiltInActionEnum action) {
        this.action = action;
    }

}
