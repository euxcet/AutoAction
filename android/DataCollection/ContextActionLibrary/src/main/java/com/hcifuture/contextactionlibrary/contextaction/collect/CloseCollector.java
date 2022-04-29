package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class CloseCollector extends BaseCollector {

    private CompletableFuture<List<CollectorResult>> FutureIMU;
//    private CompletableFuture<List<CollectorResult>> FutureNon;
    private LogCollector logCollector;


    public CloseCollector(Context context, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger, LogCollector CloseLogCollector) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger);
        logCollector = CloseLogCollector;
    }


    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onAction(ActionResult action) {
        if (action.getAction().equals("Close")) {
            FutureIMU = clickTrigger.trigger(Collections.singletonList(CollectorManager.CollectorType.IMU), new TriggerConfig());
//            FutureNon = clickTrigger.trigger(Collections.singletonList(CollectorManager.CollectorType.NonIMU), new TriggerConfig());
            long time = System.currentTimeMillis();
            if(FutureIMU !=null){
                String name = action.getAction();
                String commit = action.getAction() + ":" + action.getReason() + " " + action.getTimestamp()+" "+time;
                if(FutureIMU.isDone()) {
                    try {
                        upload(FutureIMU.get().get(0), name, commit);
//                        upload(FutureNon.get().get(0), name, commit);
//                       upload(FutureIMU.get().get(0), name, commit, time);
//                       upload(FutureNon.get().get(0), name, commit, time);
                    } catch (ExecutionException | InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                else {
                    FutureIMU.whenComplete((v, e) -> upload(v.get(0), name, commit));
//                    FutureNon.whenComplete((v, e) -> upload(v.get(0), name, commit));
                }
            }
            if (clickTrigger != null && scheduledExecutorService != null) {
                futureList.add(scheduledExecutorService.schedule(() -> {
                    try {
                        triggerAndUpload(logCollector, new TriggerConfig(), "Close", "time: "+time)
                                .thenAccept(v -> logCollector.eraseLog(v.getLogLength()));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }, 20000L, TimeUnit.MILLISECONDS));
            }

        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onContext(ContextResult context) {
    }
}
