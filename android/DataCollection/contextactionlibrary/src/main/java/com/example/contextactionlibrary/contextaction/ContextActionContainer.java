package com.example.contextactionlibrary.contextaction;

import android.app.Notification;
import android.content.Context;
import android.content.Intent;
import android.hardware.SensorManager;
import android.util.Log;

import com.example.contextactionlibrary.BuildConfig;
import com.example.contextactionlibrary.contextaction.action.ActionBase;
import com.example.contextactionlibrary.contextaction.action.ActionListener;
import com.example.contextactionlibrary.contextaction.action.KnockAction;
import com.example.contextactionlibrary.contextaction.action.TapTapAction;
import com.example.contextactionlibrary.contextaction.action.ActionConfig;
import com.example.contextactionlibrary.contextaction.context.ProximityContext;
import com.example.contextactionlibrary.data.AlwaysOnSensorManager;
import com.example.contextactionlibrary.data.ProxSensorManager;
import com.example.contextactionlibrary.model.NcnnInstance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class ContextActionContainer {
    private AlwaysOnSensorManager alwaysOnSensorManager;
    private ProxSensorManager proxSensorManager;

//    private TapTapAction taptapAction;
//    private KnockAction knockAction;
    private ProximityContext proximityContext;

    private Context mContext;

    private ThreadPoolExecutor executor;

    private List<ActionBase> actions;

    private boolean fromDex = false;
    private List<String> dexActions;

    public ContextActionContainer(Context context, List<String> actions, boolean fromDex) {
        this.mContext = context;
        this.dexActions = actions;
        this.actions = new ArrayList<>();
        this.fromDex = fromDex;
        this.executor = new ThreadPoolExecutor(1,
                1,
                1000, TimeUnit.MILLISECONDS,
                new LinkedBlockingDeque<>(2),
                new ThreadPoolExecutor.DiscardOldestPolicy());

        NcnnInstance.init(context,
                BuildConfig.SAVE_PATH + "best.param",
                BuildConfig.SAVE_PATH + "best.bin",
                4,
                128,
                6,
                1,
                2);
    }

    public ContextActionContainer(Context context, List<ActionBase> actions) {
        this.mContext = context;
        this.actions = actions;
        this.executor = new ThreadPoolExecutor(1,
                1,
                1000, TimeUnit.MILLISECONDS,
                new LinkedBlockingDeque<>(2),
                new ThreadPoolExecutor.DiscardOldestPolicy());

        NcnnInstance.init(context,
                BuildConfig.SAVE_PATH + "best.param",
                BuildConfig.SAVE_PATH + "best.bin",
                4,
                128,
                6,
                1,
                2);
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        stop();
    }

    public void start() {
        initialize();

        startSensor();
        startAction();
        startContext();

        monitorAction();
        monitorContext();
    }

    public void stop() {
        stopSensor();
        stopAction();
        stopContext();
    }

    private void initializeDexActions() {
        for(String name: dexActions) {
            if (name.equals("TapTap")) {
                ActionConfig taptapConfig = new ActionConfig();
                taptapConfig.putValue("SeqLength", 50);
                TapTapAction taptapAction = new TapTapAction(mContext, taptapConfig, (actionBase, action) -> {
                    Log.e("Action", "TapTap");
                    Intent intent = new Intent();
                    intent.setAction("contextactionlibrary");
                    intent.putExtra("Action", "TapTap");
                    mContext.sendBroadcast(intent);
                });
                actions.add(taptapAction);
            }
        }
    }

    private void initialize() {
        if (fromDex) {
            initializeDexActions();
        }

        // init sensor
        alwaysOnSensorManager = new AlwaysOnSensorManager(mContext,
                SensorManager.SENSOR_DELAY_FASTEST,
                "AlwaysOnSensorManager",
                actions
        );

        proxSensorManager = new ProxSensorManager(mContext, SensorManager.SENSOR_DELAY_FASTEST, "ProxSensorManager");

        // init context
        proximityContext = new ProximityContext(mContext,
                (contextBase, context) -> updateContextAction(context),
                0,
                new String[]{"None", "Proximity"});
    }

    private void startSensor() {
        alwaysOnSensorManager.start();
    }

    private void stopSensor() {
        alwaysOnSensorManager.stop();
    }

    private void startAction() {
        for(ActionBase action: actions) {
            action.start();
        }
        /*
        taptapAction.start();
        knockAction.start();
         */
    }

    private void stopAction() {
        for(ActionBase action: actions) {
            action.stop();
        }
        /*
        taptapAction.stop();
        knockAction.stop();
        */
    }

    private void startContext() {
        proximityContext.start();
    }

    private void stopContext() {
        proximityContext.stop();
    }

    private void monitorAction() {
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                executor.execute(() -> {
                    for(ActionBase action: actions) {
                        action.getAction();
                    }
                    /*
                    taptapAction.getAction();
                    knockAction.getAction();
                     */
                });
            }
        }, 5000, 5);
    }

    private void monitorContext() {
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                proximityContext.getContext();
            }
        }, 5000, 1000);
    }

    private void updateContextAction(String type) {
        if (type.equals("None"))
            return;
        switch (type) {
            case "TapTap":
                proxSensorManager.start();
                proxSensorManager.stopLater(3000);
                break;
            case "Proximity":
                break;
            default:
                break;
        }
    }
}
