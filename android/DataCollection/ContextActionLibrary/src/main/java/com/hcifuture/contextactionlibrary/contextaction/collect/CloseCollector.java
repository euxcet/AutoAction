package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
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

public class CloseCollector extends BaseCollector {

    private CompletableFuture<List<CollectorResult>> FutureIMU;
    private CompletableFuture<List<CollectorResult>> FutureNon;

    public CloseCollector(Context context, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger);
    }


    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onAction(ActionResult action) {
        if (action.getAction().equals("Close")) {
            FutureIMU = clickTrigger.trigger(Collections.singletonList(CollectorManager.CollectorType.IMU), new TriggerConfig());
            FutureNon = clickTrigger.trigger(Collections.singletonList(CollectorManager.CollectorType.NonIMU), new TriggerConfig());
            long time = System.currentTimeMillis();
            if(FutureIMU !=null && FutureNon!=null ){
               String name = action.getAction();
               String commit = action.getAction() + ":" + action.getReason() + " " + action.getTimestamp();
               if(FutureIMU.isDone() && FutureNon!=null) {
                   try {
                       upload(FutureIMU.get().get(0), name, commit, time);
                       upload(FutureNon.get().get(0), name, commit, time);
                   } catch (ExecutionException | InterruptedException e) {
                       e.printStackTrace();
                   }
               }
               else {
                   FutureIMU.whenComplete((v, e) -> upload(v.get(0), name, commit, time));
                   FutureNon.whenComplete((v, e) -> upload(v.get(0), name, commit, time));
               }
           }
//            FutureIMU = clickTrigger.trigger(Collections.singletonList(CollectorManager.CollectorType.NonIMU), new TriggerConfig());
//            if(FutureIMU !=null){
//                String name = action.getAction();
//                String commit = action.getAction() + ":" + action.getReason() + " " + action.getTimestamp();
//                if(FutureIMU.isDone()) {
//                    try {
//                        upload(FutureIMU.get().get(0), name, commit);
//                    } catch (ExecutionException | InterruptedException e) {
//                        e.printStackTrace();
//                    }
//                }
//                else {
//                    FutureIMU.whenComplete((v, e) -> upload(v.get(0), name, commit));
//                }
//            }
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onContext(ContextResult context) {
    }
}
