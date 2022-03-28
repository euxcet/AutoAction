package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.util.Log;

import com.hcifuture.contextactionlibrary.collect.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.utils.NetworkUtils;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;
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
        if (clickTrigger != null) {
            new Timer().schedule(new TimerTask() {
                @Override
                public void run() {
                    File imuFile = new File(clickTrigger.getRecentIMUPath());
                    Log.e("TapTapCollector", imuFile.getAbsolutePath());
                    NetworkUtils.uploadCollectedData(mContext,
                            imuFile,
                            0,
                            "TapTap",
                            getMacMoreThanM(),
                            System.currentTimeMillis(),
                            "Commit",
                            new StringCallback() {
                                @Override
                                public void onSuccess(Response<String> response) {
                                    Log.e("TapTapCollector", "Success");
                                }
                            });
                }
            }, 20000);
        }
    }

    @Override
    public void onContext(ContextResult context) {

    }
}
