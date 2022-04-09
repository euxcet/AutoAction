package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.util.Log;

import com.hcifuture.contextactionlibrary.collect.collector.Collector;
import com.hcifuture.contextactionlibrary.collect.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.contextaction.context.informational.InformationalContext;
import com.hcifuture.contextactionlibrary.utils.FileUtils;
import com.hcifuture.contextactionlibrary.utils.NetworkUtils;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class InformationalContextCollector extends BaseCollector {

    String informationalFile;
    String informationalTmpFolder;
    public InformationalContextCollector(Context context, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger);

        informationalFile = mContext.getExternalMediaDirs()[0].getAbsolutePath()+"/informational/taskLog.txt";
        informationalTmpFolder = mContext.getExternalMediaDirs()[0].getAbsolutePath()+"/informational/tmp/";
    }

    @Override
    public void onAction(ActionResult action) {
        upload(action);
    }

    public void upload(ActionResult action)
    {
        File nowFile = new File(informationalFile);
        File tmpFolder = new File(informationalTmpFolder);
        File tmpFile = new File(informationalTmpFolder+System.currentTimeMillis()+"_taskLog.txt");
        FileUtils.copy(nowFile,tmpFile);

        futureList.add(scheduledExecutorService.schedule(() -> {
            File[] tmpFiles =tmpFolder.listFiles();
            for(File file:tmpFiles) {
                Log.e("InformationalCollector", file.getAbsolutePath());
                NetworkUtils.uploadCollectedData(mContext,
                        file,
                        0,
                        action.getAction(),
                        getMacMoreThanM(),
                        System.currentTimeMillis(),
                        action.getAction() + ":" + action.getReason() + ":" + action.getTimestamp(),
                        new StringCallback() {
                            @Override
                            public void onSuccess(Response<String> response) {
                                if(file.exists())
                                    file.delete();
                            }
                        }
                );
            }
        }, 2000L, TimeUnit.MILLISECONDS));
    }


    @Override
    public void onContext(ContextResult context) {

    }


}
