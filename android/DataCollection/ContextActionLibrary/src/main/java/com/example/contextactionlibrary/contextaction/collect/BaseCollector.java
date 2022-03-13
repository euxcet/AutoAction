package com.example.contextactionlibrary.contextaction.collect;

import android.content.Context;

import com.example.contextactionlibrary.collect.trigger.ClickTrigger;
import com.example.ncnnlibrary.communicate.listener.RequestListener;
import com.example.ncnnlibrary.communicate.result.ActionResult;
import com.example.ncnnlibrary.communicate.result.ContextResult;

public abstract class BaseCollector {
    protected Context mContext;
    protected RequestListener requestListener;
    protected ClickTrigger clickTrigger;

    public BaseCollector(Context context, RequestListener requestListener, ClickTrigger clickTrigger) {
        this.mContext = context;
        this.requestListener = requestListener;
        this.clickTrigger = clickTrigger;
    }

    public abstract void onAction(ActionResult action);

    public abstract void onContext(ContextResult context);
}
