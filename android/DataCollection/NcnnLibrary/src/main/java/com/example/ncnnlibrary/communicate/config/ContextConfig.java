package com.example.ncnnlibrary.communicate.config;

import com.example.ncnnlibrary.communicate.BuiltInContextEnum;

public class ContextConfig extends Config {
    private BuiltInContextEnum context;

    public BuiltInContextEnum getContext() {
        return context;
    }

    public void setContext(BuiltInContextEnum context) {
        this.context = context;
    }
}
