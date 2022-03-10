package com.example.contextactionlibrary.contextaction;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;

import com.example.contextactionlibrary.BuildConfig;
import com.example.contextactionlibrary.contextaction.action.ActionBase;
import com.example.contextactionlibrary.contextaction.action.TapTapAction;
import com.example.contextactionlibrary.contextaction.context.ContextBase;
import com.example.contextactionlibrary.contextaction.context.ProximityContext;
import com.example.contextactionlibrary.data.IMUSensorManager;
import com.example.contextactionlibrary.data.ProximitySensorManager;
import com.example.contextactionlibrary.model.NcnnInstance;
import com.example.ncnnlibrary.communicate.BuiltInContextEnum;
import com.example.ncnnlibrary.communicate.config.ActionConfig;
import com.example.ncnnlibrary.communicate.config.ContextConfig;
import com.example.ncnnlibrary.communicate.listener.ActionListener;
import com.example.ncnnlibrary.communicate.BuiltInActionEnum;
import com.example.ncnnlibrary.communicate.SensorType;
import com.example.ncnnlibrary.communicate.listener.ContextListener;

import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class ContextActionContainer {
    private IMUSensorManager imuSensorManager;
    private ProximitySensorManager proximitySensorManager;

    private Context mContext;

    private ThreadPoolExecutor executor;

    private List<ActionBase> actions;
    private List<ContextBase> contexts;

    private boolean fromDex = false;
    private List<ActionConfig> actionConfig;
    private List<ContextConfig> contextConfig;
    private ActionListener actionListener;
    private ContextListener contextListener;

    public ContextActionContainer(Context context, List<ActionBase> actions, List<ContextBase> contexts) {
        this.mContext = context;
        this.actions = actions;
        this.contexts = contexts;
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

    public ContextActionContainer(Context context,
                                  List<ActionConfig> actionConfig, ActionListener actionListener,
                                  List<ContextConfig> contextConfig, ContextListener contextListener,
                                  boolean fromDex) {
        this(context, new ArrayList<>(), new ArrayList<>());
        this.actionConfig = actionConfig;
        this.actionListener = actionListener;
        this.contextConfig = contextConfig;
        this.contextListener = contextListener;
        this.fromDex = fromDex;
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

    private void initializeDexContextActions() {
        for (int i = 0; i < actionConfig.size(); i++) {
            ActionConfig config = actionConfig.get(i);
            if (config.getAction() == BuiltInActionEnum.TapTap) {
                TapTapAction tapTapAction = new TapTapAction(mContext, config, actionListener);
                actions.add(tapTapAction);
            }
        }
        for (int i = 0; i < contextConfig.size(); i++) {
            ContextConfig config = contextConfig.get(i);
            if (config.getContext() == BuiltInContextEnum.Proximity) {
                ProximityContext proximityContext = new ProximityContext(mContext, config, contextListener);
                contexts.add(proximityContext);
            }
        }
    }

    private List<ActionBase> selectBySensorTypeAction(List<ActionBase> actions, SensorType sensorType) {
        List<ActionBase> result = new ArrayList<>();
        for (ActionBase action: actions) {
            if (action.getConfig().getSensorType().contains(sensorType)) {
                result.add(action);
            }
        }
        return result;
    }

    private List<ContextBase> selectBySensorTypeContext(List<ContextBase> contexts, SensorType sensorType) {
        List<ContextBase> result = new ArrayList<>();
        for (ContextBase context: contexts) {
            if (context.getConfig().getSensorType().contains(sensorType)) {
                result.add(context);
            }
        }
        return result;
    }

    private void initialize() {
        if (fromDex) {
            initializeDexContextActions();
        }

        // init sensor
        imuSensorManager = new IMUSensorManager(mContext,
                "AlwaysOnSensorManager",
                selectBySensorTypeAction(actions, SensorType.IMU),
                selectBySensorTypeContext(contexts, SensorType.IMU),
                SensorManager.SENSOR_DELAY_FASTEST
        );

        proximitySensorManager = new ProximitySensorManager(mContext,
                "ProximitySensorManager",
                selectBySensorTypeAction(actions, SensorType.PROXIMITY),
                selectBySensorTypeContext(contexts, SensorType.PROXIMITY),
                SensorManager.SENSOR_DELAY_FASTEST
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
        imuSensorManager.start();
    }

    private void stopSensor() {
        imuSensorManager.stop();
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
                if (imuSensorManager != null) {
                    imuSensorManager.onSensorChangedDex(event);
                }
            case Sensor.TYPE_PROXIMITY:
                if (proximitySensorManager != null) {
                    proximitySensorManager.onSensorChangedDex(event);
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
                proximitySensorManager.start();
                proximitySensorManager.stopLater(3000);
                break;
            case "Proximity":
                break;
            default:
                break;
        }
    }
}
