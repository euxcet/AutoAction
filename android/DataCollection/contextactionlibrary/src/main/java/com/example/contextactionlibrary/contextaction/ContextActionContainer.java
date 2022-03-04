package com.example.contextactionlibrary.contextaction;

import android.content.Context;
import android.content.pm.ShortcutManager;
import android.hardware.SensorManager;
import android.util.Log;

import com.example.contextactionlibrary.contextaction.action.KnockAction;
import com.example.contextactionlibrary.contextaction.action.TapTapAction;
import com.example.contextactionlibrary.contextaction.context.ProximityContext;
import com.example.contextactionlibrary.data.AlwaysOnSensorManager;
import com.example.contextactionlibrary.data.ProxSensorManager;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class ContextActionContainer {
    private AlwaysOnSensorManager alwaysOnSensorManager;
    private ProxSensorManager proxSensorManager;

    // private TapTapAction taptapAction;
    private KnockAction knockAction;
    private ProximityContext proximityContext;

    private ShortcutManager THUPatShortcutManager;
    private Context mContext;

    private ThreadPoolExecutor executor;

    public ContextActionContainer() {

    }

    public ContextActionContainer(String s) {
        Log.e("ContextAction", "string constructor");
    }

    public ContextActionContainer(Context context) {
        this.mContext = context;
        this.executor = new ThreadPoolExecutor(1,
                1,
                1000, TimeUnit.MILLISECONDS,
                new LinkedBlockingDeque<>(2),
                new ThreadPoolExecutor.DiscardOldestPolicy());
        Log.e("ContextAction", "context constructor");
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        stop();
    }

    public void start() {
        Log.e("ContextAction", "start?");
        initialize();

        startSensor();
        startAction();
        startContext();

        monitorAction();
        monitorContext();
        Log.e("ContextAction", "start");
    }

    public void stop() {
        Log.e("ContextAction", "stop?");
        stopSensor();
        stopAction();
        stopContext();
        Log.e("ContextAction", "stop");
    }

    private void initialize() {
        // init action
        /*
        taptapAction = new TapTapAction(mContext,
                (actionBase, action) -> updateContextAction(action),
                50,
                new String[]{"None", "TapTap"});
         */

        knockAction = new KnockAction(mContext,
                (actionBase, action) -> updateContextAction(action),
                128,
                new String[]{"None", "Knock"});

        // init sensor
        alwaysOnSensorManager = new AlwaysOnSensorManager(mContext,
                SensorManager.SENSOR_DELAY_FASTEST,
                "AlwaysOnSensorManager",
                Arrays.asList(knockAction)
                // Arrays.asList(taptapAction, knockAction)
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
        // taptapAction.start();
        knockAction.start();
    }

    private void stopAction() {
        // taptapAction.stop();
        knockAction.stop();
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
                // taptapAction.getAction();
                executor.execute(() -> knockAction.getAction());
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
