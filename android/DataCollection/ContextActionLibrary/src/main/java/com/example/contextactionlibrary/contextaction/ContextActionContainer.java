package com.example.contextactionlibrary.contextaction;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;
import android.util.Log;

import com.example.contextactionlibrary.BuildConfig;
import com.example.contextactionlibrary.contextaction.action.ActionBase;
import com.example.contextactionlibrary.contextaction.action.TapTapAction;
import com.example.contextactionlibrary.contextaction.context.ProximityContext;
import com.example.contextactionlibrary.data.AlwaysOnSensorManager;
import com.example.contextactionlibrary.data.ProxSensorManager;
import com.example.contextactionlibrary.model.NcnnInstance;
import com.example.ncnnlibrary.communicate.ActionConfig;
import com.example.ncnnlibrary.communicate.ActionListener;
import com.example.ncnnlibrary.communicate.BuiltInActionEnum;

import java.util.ArrayList;
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
    private List<ActionConfig> dexActionConfig;
    private ActionListener dexActionListener;

    public ContextActionContainer(Context context, List<ActionConfig> actionConfig, ActionListener actionListener, boolean fromDex) {
        this.mContext = context;
        this.dexActionConfig = actionConfig;
        this.dexActionListener = actionListener;
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

        // startSensor();
        startAction();
        startContext();

        monitorAction();
        monitorContext();
    }

    public void stop() {
        // stopSensor();
        stopAction();
        stopContext();
    }

    private void initializeDexActions() {
        for (int i = 0; i < dexActionConfig.size(); i++) {
            ActionConfig config = dexActionConfig.get(i);
            if (config.getAction() == BuiltInActionEnum.TapTap) {
                TapTapAction tapTapAction = new TapTapAction(mContext, config, dexActionListener);
                actions.add(tapTapAction);
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

        proxSensorManager = new ProxSensorManager(mContext,
                SensorManager.SENSOR_DELAY_FASTEST,
                "ProxSensorManager"
        );

        // init context
        /*
        proximityContext = new ProximityContext(mContext,
                (contextBase, context) -> updateContextAction(context),
                0,
                new String[]{"None", "Proximity"});

         */
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
        // proximityContext.start();
    }

    private void stopContext() {
        // proximityContext.stop();
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
                // proximityContext.getContext();
            }
        }, 5000, 1000);
    }

    public void onSensorChangedDex(SensorEvent event) {
        int type = event.sensor.getType();
        switch (type) {
            case Sensor.TYPE_GYROSCOPE:
            case Sensor.TYPE_ACCELEROMETER:
                if (alwaysOnSensorManager != null) {
                    alwaysOnSensorManager.onSensorChangedDex(event);
                }
            case Sensor.TYPE_PROXIMITY:
                if (proxSensorManager != null) {
                    proxSensorManager.onSensorChangedDex(event);
                }
            default:
                break;
        }
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
