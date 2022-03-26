package com.hcifuture.contextactionlibrary.contextaction;

import android.content.Context;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;

import com.hcifuture.contextactionlibrary.BuildConfig;
import com.hcifuture.contextactionlibrary.collect.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.collect.trigger.Trigger;
import com.hcifuture.contextactionlibrary.contextaction.action.BaseAction;
import com.hcifuture.contextactionlibrary.contextaction.action.TapTapAction;
import com.hcifuture.contextactionlibrary.contextaction.collect.BaseCollector;
import com.hcifuture.contextactionlibrary.contextaction.collect.TapTapCollector;
import com.hcifuture.contextactionlibrary.contextaction.context.BaseContext;
import com.hcifuture.contextactionlibrary.contextaction.context.informational.InformationalContext;
import com.hcifuture.contextactionlibrary.contextaction.context.physical.ProximityContext;
import com.hcifuture.contextactionlibrary.contextaction.context.physical.TableContext;
import com.hcifuture.contextactionlibrary.data.AccessibilityEventManager;
import com.hcifuture.contextactionlibrary.data.BroadcastEventManager;
import com.hcifuture.contextactionlibrary.data.IMUSensorManager;
import com.hcifuture.contextactionlibrary.data.BaseSensorManager;
import com.hcifuture.contextactionlibrary.data.ProximitySensorManager;
import com.hcifuture.contextactionlibrary.model.NcnnInstance;
import com.hcifuture.shared.communicate.BuiltInContextEnum;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.shared.communicate.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.BuiltInActionEnum;
import com.hcifuture.shared.communicate.SensorType;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class ContextActionContainer implements ActionListener, ContextListener {
    // private IMUSensorManager imuSensorManager;
    // private ProximitySensorManager proximitySensorManager;

    private Context mContext;

    private ThreadPoolExecutor executor;

    private List<BaseAction> actions;
    private List<BaseContext> contexts;

    private List<BaseSensorManager> sensorManagers;

    private boolean fromDex = false;
    private boolean openSensor = true;
    private List<ActionConfig> actionConfig;
    private List<ContextConfig> contextConfig;
    private ActionListener actionListener;
    private ContextListener contextListener;
    private RequestListener requestListener;

    private ClickTrigger clickTrigger;

    private List<BaseCollector> collectors;

    public ContextActionContainer(Context context, List<BaseAction> actions, List<BaseContext> contexts, RequestListener requestListener) {
        this.mContext = context;
        this.actions = actions;
        this.contexts = contexts;
        this.executor = new ThreadPoolExecutor(1,
                1,
                1000, TimeUnit.MILLISECONDS,
                new LinkedBlockingDeque<>(2),
                new ThreadPoolExecutor.DiscardOldestPolicy());
        this.sensorManagers = new ArrayList<>();

        if (NcnnInstance.getInstance() == null) {
            NcnnInstance.init(context,
                    BuildConfig.SAVE_PATH + "best.param",
                    BuildConfig.SAVE_PATH + "best.bin",
                    4,
                    128,
                    6,
                    1,
                    2);
        }

        clickTrigger = new ClickTrigger(context, Arrays.asList(Trigger.CollectorType.CompleteIMU, Trigger.CollectorType.Bluetooth));
        // clickTrigger = new ClickTrigger(context, Arrays.asList(Trigger.CollectorType.CompleteIMU));
        collectors = Arrays.asList(new TapTapCollector(context, requestListener, clickTrigger));

        scheduleCleanData();
    }

    public ContextActionContainer(Context context,
                                  List<ActionConfig> actionConfig, ActionListener actionListener,
                                  List<ContextConfig> contextConfig, ContextListener contextListener,
                                  RequestListener requestListener,
                                  boolean fromDex, boolean openSensor) {
        this(context, new ArrayList<>(), new ArrayList<>(), requestListener);
        this.actionConfig = actionConfig;
        this.actionListener = actionListener;
        this.contextConfig = contextConfig;
        this.contextListener = contextListener;
        this.requestListener = requestListener;
        this.fromDex = fromDex;
        this.openSensor = openSensor;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        stop();
    }

    public void start() {
        initialize();

        if (openSensor) {
            for (BaseSensorManager sensorManager: sensorManagers) {
                sensorManager.start();
            }
        }

        for (BaseAction action: actions) {
            action.start();
        }
        for (BaseContext context: contexts) {
            context.start();
        }
        monitorAction();
        monitorContext();
    }

    public void stop() {
        if (openSensor) {
            for (BaseSensorManager sensorManager: sensorManagers) {
                sensorManager.stop();
            }
        }
        for (BaseAction action: actions) {
            action.stop();
        }
        for (BaseContext context: contexts) {
            context.stop();
        }
    }

    private List<BaseAction> selectBySensorTypeAction(List<BaseAction> actions, SensorType sensorType) {
        List<BaseAction> result = new ArrayList<>();
        for (BaseAction action: actions) {
            if (action.getConfig().getSensorType().contains(sensorType)) {
                result.add(action);
            }
        }
        return result;
    }

    private List<BaseContext> selectBySensorTypeContext(List<BaseContext> contexts, SensorType sensorType) {
        List<BaseContext> result = new ArrayList<>();
        for (BaseContext context: contexts) {
            if (context.getConfig().getSensorType().contains(sensorType)) {
                result.add(context);
            }
        }
        return result;
    }

    private void initialize() {
        if (fromDex) {
            for (int i = 0; i < actionConfig.size(); i++) {
                ActionConfig config = actionConfig.get(i);
                if (config.getAction() == BuiltInActionEnum.TapTap) {
                    TapTapAction tapTapAction = new TapTapAction(mContext, config, requestListener, Arrays.asList(this, actionListener));
                    actions.add(tapTapAction);
                }
            }
            for (int i = 0; i < contextConfig.size(); i++) {
                ContextConfig config = contextConfig.get(i);
                if (config.getContext() == BuiltInContextEnum.Proximity) {
                    ProximityContext proximityContext = new ProximityContext(mContext, config, requestListener, Arrays.asList(this, contextListener));
                    contexts.add(proximityContext);
                } else if (config.getContext() == BuiltInContextEnum.Table) {
                    TableContext tableContext = new TableContext(mContext, config, requestListener, Arrays.asList(this, contextListener));
                    contexts.add(tableContext);
                } else if (config.getContext() == BuiltInContextEnum.Informational) {
                    InformationalContext informationalContext = new InformationalContext(mContext, config, requestListener, Arrays.asList(this, contextListener));
                    contexts.add(informationalContext);
                }
            }
        }

        // init sensor
        sensorManagers.add(new IMUSensorManager(mContext,
                "IMUSensorManager",
                selectBySensorTypeAction(actions, SensorType.IMU),
                selectBySensorTypeContext(contexts, SensorType.IMU),
                SensorManager.SENSOR_DELAY_FASTEST
        ));

        sensorManagers.add(new ProximitySensorManager(mContext,
                "ProximitySensorManager",
                selectBySensorTypeAction(actions, SensorType.PROXIMITY),
                selectBySensorTypeContext(contexts, SensorType.PROXIMITY),
                SensorManager.SENSOR_DELAY_FASTEST
        ));

        sensorManagers.add(new AccessibilityEventManager(mContext,
                "AccessibilityEventManager",
                selectBySensorTypeAction(actions, SensorType.ACCESSIBILITY),
                selectBySensorTypeContext(contexts, SensorType.ACCESSIBILITY)
        ));

        sensorManagers.add(new BroadcastEventManager(mContext,
                "BroadcastEventManager",
                selectBySensorTypeAction(actions, SensorType.BROADCAST),
                selectBySensorTypeContext(contexts, SensorType.BROADCAST)
        ));
    }

    private void monitorAction() {
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                executor.execute(() -> {
                    for (BaseAction action: actions) {
                        action.getAction();
                    }
                });
            }
        }, 5000, 20);
    }

    private void monitorContext() {
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                executor.execute(() -> {
                    for (BaseContext context: contexts) {
                        context.getContext();
                    }
                });
            }
        }, 5000, 1000);
    }

    public void onSensorChangedDex(SensorEvent event) {
        int type = event.sensor.getType();
        for (BaseSensorManager sensorManager: sensorManagers) {
            if (sensorManager.getSensorTypeList() != null && sensorManager.getSensorTypeList().contains(type)) {
                sensorManager.onSensorChangedDex(event);
            }
        }
    }

    public void onAccessibilityEventDex(AccessibilityEvent event) {
        for (BaseSensorManager sensorManager: sensorManagers) {
            sensorManager.onAccessibilityEventDex(event);
        }
    }

    public void onBroadcastEventDex(BroadcastEvent event) {
        for (BaseSensorManager sensorManager: sensorManagers) {
            sensorManager.onBroadcastEventDex(event);
        }
    }

    /*
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
     */

    private void scheduleCleanData() {
        Calendar calendar = Calendar.getInstance();
        calendar.set(calendar.get(Calendar.YEAR), calendar.get(Calendar.MONTH), calendar.get(Calendar.DAY_OF_MONTH),3, 0, 0);
        Date date = calendar.getTime();
        if (date.before(new Date())) {
            calendar.add(Calendar.DATE, 1);
            date = calendar.getTime();
        }
        Timer timer= new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                clickTrigger.cleanData();
                Log.e("TapTapCollector", "triggered");
            }
        }, date, 24 * 60 * 60 * 1000);
    }

    @Override
    public void onAction(ActionResult action) {
        Log.e("TapTapCollector", "On Action  " + action.getAction());
        if (action.getAction().equals("TapTap")) {
            clickTrigger.trigger();
        }
        if (collectors != null) {
            for (BaseCollector collector: collectors) {
                collector.onAction(action);
            }
        }
    }

    @Override
    public void onContext(ContextResult context) {
        if (collectors != null) {
            for (BaseCollector collector: collectors) {
                collector.onContext(context);
            }
        }
    }
}
