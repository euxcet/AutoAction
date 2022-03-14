package com.example.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.util.Log;

import com.example.contextactionlibrary.collect.trigger.ClickTrigger;
import com.example.contextactionlibrary.utils.NetworkUtils;
import com.example.ncnnlibrary.communicate.listener.RequestListener;
import com.example.ncnnlibrary.communicate.result.ActionResult;
import com.example.ncnnlibrary.communicate.result.ContextResult;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.Timer;
import java.util.TimerTask;

public class TapTapCollector extends BaseCollector {
    public TapTapCollector(Context context, RequestListener requestListener, ClickTrigger clickTrigger) {
        super(context, requestListener, clickTrigger);
    }

    @Override
    public void onAction(ActionResult action) {
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                File imuFile = new File(clickTrigger.getRecentIMUPath());
                Log.e("TapTapCollector", imuFile.getAbsolutePath());
                NetworkUtils.uploadCollectedData(mContext, imuFile, 0, "TapTap", getMacMoreThanM(), System.currentTimeMillis(),"Commit", new StringCallback() {
                    @Override
                    public void onSuccess(Response<String> response) {

                    }
                });
            }
        }, 65000);
    }

    @Override
    public void onContext(ContextResult context) {

    }
}
