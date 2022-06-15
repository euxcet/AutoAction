package com.hcifuture.contextactionlibrary.contextaction;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.ContentObserver;
import android.hardware.SensorEvent;
import android.net.Uri;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.Settings;
import android.util.Log;
import android.view.KeyEvent;
import android.view.accessibility.AccessibilityEvent;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;

//import com.amap.api.services.core.ServiceSettings;
import com.google.gson.Gson;
import com.hcifuture.contextactionlibrary.contextaction.action.MotionAction;
import com.hcifuture.contextactionlibrary.contextaction.collect.CloseCollector;
import com.hcifuture.contextactionlibrary.contextaction.collect.ConfigCollector;
import com.hcifuture.contextactionlibrary.contextaction.collect.FlipCollector;
import com.hcifuture.contextactionlibrary.contextaction.collect.InformationalContextCollector;
import com.hcifuture.contextactionlibrary.contextaction.collect.TapTapCollector;
import com.hcifuture.contextactionlibrary.contextaction.collect.TimedCollector;
import com.hcifuture.contextactionlibrary.contextaction.context.ConfigContext;
import com.hcifuture.contextactionlibrary.contextaction.context.informational.InformationalContext;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorStatusHolder;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.distributor.DataDistributor;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.contextaction.action.BaseAction;
import com.hcifuture.contextactionlibrary.contextaction.action.CloseAction;
import com.hcifuture.contextactionlibrary.contextaction.action.FlipAction;
import com.hcifuture.contextactionlibrary.contextaction.action.PocketAction;
import com.hcifuture.contextactionlibrary.contextaction.action.TapTapAction;
import com.hcifuture.contextactionlibrary.contextaction.action.TopTapAction;
import com.hcifuture.contextactionlibrary.contextaction.collect.BaseCollector;
import com.hcifuture.contextactionlibrary.contextaction.context.BaseContext;
import com.hcifuture.contextactionlibrary.contextaction.context.physical.ProximityContext;
import com.hcifuture.contextactionlibrary.contextaction.context.physical.TableContext;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.sensor.uploader.Uploader;
import com.hcifuture.contextactionlibrary.status.Heart;
import com.hcifuture.contextactionlibrary.utils.FileSaver;
import com.hcifuture.contextactionlibrary.utils.FileUtils;
import com.hcifuture.contextactionlibrary.utils.JSONUtils;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.contextactionlibrary.contextaction.event.BroadcastEvent;
import com.hcifuture.shared.communicate.config.RequestConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.SensorType;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;
import com.hcifuture.shared.communicate.result.Result;
import com.hcifuture.shared.communicate.status.Heartbeat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

@RequiresApi(api = Build.VERSION_CODES.N)
public class ContextActionContainer implements ActionListener, ContextListener {
    private final Context mContext;

    // private ThreadPoolExecutor executor;

    private final List<BaseAction> actions;
    private final List<BaseContext> contexts;

    // private List<BaseSensorManager> sensorManagers;

    private boolean fromDex = false;
    private boolean openSensor = false;
    private ActionListener actionListener;
    private ContextListener contextListener;
    private RequestListener requestListener;

    private ClickTrigger clickTrigger;
    private Uploader uploader;

    private List<BaseCollector> collectors;

    private ScheduledExecutorService scheduledExecutorService;
    private final List<ScheduledFuture<?>> futureList;
    private HandlerThread handlerThread;
    private Handler handler;

    private ScheduledFuture<?> actionFuture;
    private ScheduledFuture<?> contextFuture;

    private CollectorManager collectorManager;
    private DataDistributor dataDistributor;

    private static String SAVE_PATH;

    private final AtomicInteger mContextActionIDCounter = new AtomicInteger(0);
    LogCollector contextActionLogCollector;

    private CustomBroadcastReceiver mBroadcastReceiver;
    private CustomContentObserver mContentObserver;
    private final List<Uri> mRegURIs;
    private final Lock contextLock = new ReentrantLock();

    // listening
    private final Uri [] listenedURIs = {
            Settings.System.CONTENT_URI,
            Settings.Global.CONTENT_URI,
    };
    private final String [] listenedActions = {
            Intent.ACTION_AIRPLANE_MODE_CHANGED,
            Intent.ACTION_APPLICATION_RESTRICTIONS_CHANGED,
            Intent.ACTION_BATTERY_LOW,
            Intent.ACTION_BATTERY_OKAY,
            Intent.ACTION_BOOT_COMPLETED,
            Intent.ACTION_CONFIGURATION_CHANGED,
            Intent.ACTION_DOCK_EVENT,
            Intent.ACTION_DREAMING_STARTED,
            Intent.ACTION_DREAMING_STOPPED,
            Intent.ACTION_EXTERNAL_APPLICATIONS_AVAILABLE,
            Intent.ACTION_EXTERNAL_APPLICATIONS_UNAVAILABLE,
            Intent.ACTION_HEADSET_PLUG,
            Intent.ACTION_INPUT_METHOD_CHANGED,
            Intent.ACTION_LOCALE_CHANGED,
            Intent.ACTION_LOCKED_BOOT_COMPLETED,
            Intent.ACTION_MEDIA_BAD_REMOVAL,
            Intent.ACTION_MEDIA_BUTTON,
            Intent.ACTION_MEDIA_CHECKING,
            Intent.ACTION_MEDIA_EJECT,
            Intent.ACTION_MEDIA_MOUNTED,
            Intent.ACTION_MEDIA_NOFS,
            Intent.ACTION_MEDIA_REMOVED,
            Intent.ACTION_MEDIA_SCANNER_FINISHED,
            Intent.ACTION_MEDIA_SCANNER_STARTED,
            Intent.ACTION_MEDIA_SHARED,
            Intent.ACTION_MEDIA_UNMOUNTABLE,
            Intent.ACTION_MEDIA_UNMOUNTED,
            Intent.ACTION_MY_PACKAGE_REPLACED,
            Intent.ACTION_PACKAGES_SUSPENDED,
            Intent.ACTION_PACKAGES_UNSUSPENDED,
            Intent.ACTION_PACKAGE_ADDED,
            Intent.ACTION_PACKAGE_CHANGED,
            Intent.ACTION_PACKAGE_DATA_CLEARED,
            Intent.ACTION_PACKAGE_FIRST_LAUNCH,
            Intent.ACTION_PACKAGE_FULLY_REMOVED,
            Intent.ACTION_PACKAGE_NEEDS_VERIFICATION,
            Intent.ACTION_PACKAGE_REMOVED,
            Intent.ACTION_PACKAGE_REPLACED,
            Intent.ACTION_PACKAGE_RESTARTED,
            Intent.ACTION_PACKAGE_VERIFIED,
            Intent.ACTION_POWER_CONNECTED,
            Intent.ACTION_POWER_DISCONNECTED,
            Intent.ACTION_PROVIDER_CHANGED,
            Intent.ACTION_REBOOT,
            Intent.ACTION_SCREEN_OFF,
            Intent.ACTION_SCREEN_ON,
            Intent.ACTION_SHUTDOWN,
            Intent.ACTION_TIMEZONE_CHANGED,
            Intent.ACTION_TIME_CHANGED,
            Intent.ACTION_UID_REMOVED,
            Intent.ACTION_USER_BACKGROUND,
            Intent.ACTION_USER_FOREGROUND,
            Intent.ACTION_USER_PRESENT,
            Intent.ACTION_USER_UNLOCKED,
            // Bluetooth related
            BluetoothDevice.ACTION_ACL_CONNECTED,
            BluetoothDevice.ACTION_ACL_DISCONNECT_REQUESTED,
            BluetoothDevice.ACTION_ACL_DISCONNECTED,
            BluetoothDevice.ACTION_ALIAS_CHANGED,
            BluetoothDevice.ACTION_BOND_STATE_CHANGED,
            BluetoothDevice.ACTION_CLASS_CHANGED,
            BluetoothDevice.ACTION_NAME_CHANGED,
            BluetoothDevice.ACTION_PAIRING_REQUEST,
            BluetoothAdapter.ACTION_DISCOVERY_STARTED,
            BluetoothAdapter.ACTION_DISCOVERY_FINISHED,
            BluetoothAdapter.ACTION_CONNECTION_STATE_CHANGED,
            BluetoothAdapter.ACTION_STATE_CHANGED,
            BluetoothAdapter.ACTION_SCAN_MODE_CHANGED,
            BluetoothAdapter.ACTION_LOCAL_NAME_CHANGED,
            // WiFi related
            WifiManager.NETWORK_STATE_CHANGED_ACTION,
            WifiManager.WIFI_STATE_CHANGED_ACTION,
            WifiManager.SCAN_RESULTS_AVAILABLE_ACTION,
            WifiManager.ACTION_WIFI_SCAN_AVAILABILITY_CHANGED
    };

    public ContextActionContainer(Context context, List<BaseAction> actions, List<BaseContext> contexts, RequestListener requestListener, String SAVE_PATH) {
        this.mContext = context;
        this.actions = actions;
        this.contexts = contexts;
        this.actionFuture = null;
        this.contextFuture = null;
        ContextActionContainer.SAVE_PATH = SAVE_PATH;
        /*
        this.executor = new ThreadPoolExecutor(1,
                1,
                1000, TimeUnit.MILLISECONDS,
                new LinkedBlockingDeque<>(2),
                new ThreadPoolExecutor.DiscardOldestPolicy());
         */
        // this.sensorManagers = new ArrayList<>();

        /*
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
         */

        this.futureList = Collections.synchronizedList(new ArrayList<>());
        mRegURIs = new ArrayList<>();
    }

    public ContextActionContainer(Context context,
                                  ActionListener actionListener, ContextListener contextListener,
                                  RequestListener requestListener,
                                  boolean fromDex, boolean openSensor, String SAVE_PATH) {
        this(context, new ArrayList<>(), new ArrayList<>(), requestListener, SAVE_PATH);
        // this.actionConfig = actionConfig;
        this.actionListener = actionListener;
        // this.contextConfig = contextConfig;
        this.contextListener = contextListener;
        this.requestListener = requestListener;
        this.fromDex = fromDex;
        this.openSensor = openSensor;
    }

    public static String getSavePath() {
        return SAVE_PATH;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        stop();
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public void start() {
        initialize();
        /*
        if (openSensor) {
            for (BaseSensorManager sensorManager: sensorManagers) {
                sensorManager.start();
            }
        }
         */

        if (dataDistributor != null) {
            dataDistributor.start();
        }

        for (BaseAction action: actions) {
            action.start();
        }
        for (BaseContext context: contexts) {
            context.start();
        }
        monitorAction();
        monitorContext();
        if (collectorManager != null) {
            collectorManager.resume();
        }
    }

    public void stop() {
        if (FileSaver.getInstance() != null) {
            FileSaver.getInstance().close();
        }
        try {
            // unregister broadcast receiver
            mContext.unregisterReceiver(mBroadcastReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            // unregister content observer
            mContext.getContentResolver().unregisterContentObserver(mContentObserver);
        } catch (Exception e) {
            e.printStackTrace();
        }
        mRegURIs.clear();

        if (dataDistributor != null) {
            dataDistributor.stop();
        }

        for (BaseAction action: actions) {
            action.stop();
        }

        for (BaseContext context: contexts) {
            context.stop();
        }

        if (collectorManager != null) {
            if (dataDistributor != null) {
                collectorManager.unregisterListener(dataDistributor);
            }
            collectorManager.pause();
            collectorManager.close();
        }
        if (uploader != null) {
            uploader.stop();
        }
        synchronized (futureList) {
            for (ScheduledFuture<?> future : futureList) {
                future.cancel(true);
            }
        }
        futureList.clear();
        scheduledExecutorService.shutdownNow();
        handlerThread.quit();
    }

    public void pause() {
        for (BaseAction action: actions) {
            action.stop();
        }
        for (BaseContext context: contexts) {
            context.stop();
        }
        if (dataDistributor != null) {
            dataDistributor.stop();
        }
        if (collectorManager != null) {
            collectorManager.pause();
        }
    }

    public void resume() {
        if (dataDistributor != null) {
            dataDistributor.start();
        }
        for (BaseAction action: actions) {
            action.start();
        }
        for (BaseContext context: contexts) {
            context.start();
        }
        if (actionFuture != null && (actionFuture.isDone() || actionFuture.isCancelled())) {
            monitorAction();
        }
        if (contextFuture != null && (contextFuture.isDone() || contextFuture.isCancelled())) {
            monitorContext();
        }
        if (collectorManager != null) {
            collectorManager.resume();
        }
    }

    private void cleanFutureList() {
        List<ScheduledFuture<?>> removed = new ArrayList<>();
        synchronized (futureList) {
            for (ScheduledFuture<?> future: futureList) {
                if (future.isCancelled() || future.isDone()) {
                    removed.add(future);
                }
            }
        }
        futureList.removeAll(removed);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void initialize() {
//        ServiceSettings.updatePrivacyShow(mContext.getApplicationContext(), true , true);
//        ServiceSettings.updatePrivacyAgree(mContext.getApplicationContext(), true);

        handlerThread = new HandlerThread("CallbackHandlerThread");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
        Collector.setHandler(handler);
        mBroadcastReceiver = new CustomBroadcastReceiver();
        mContentObserver = new CustomContentObserver(handler);

        this.scheduledExecutorService = Executors.newScheduledThreadPool(32);
        ((ScheduledThreadPoolExecutor)scheduledExecutorService).setRemoveOnCancelPolicy(true);

        this.futureList.add(scheduledExecutorService.scheduleAtFixedRate(this::cleanFutureList, 600000, 600000, TimeUnit.MILLISECONDS));
        FileSaver.initialize(scheduledExecutorService, futureList);

        this.collectorManager = new CollectorManager(mContext, Arrays.asList(
                CollectorManager.CollectorType.IMU,
                CollectorManager.CollectorType.Location,
                CollectorManager.CollectorType.Audio,
                CollectorManager.CollectorType.Bluetooth,
                CollectorManager.CollectorType.Wifi,
                CollectorManager.CollectorType.GPS,
                CollectorManager.CollectorType.NonIMU
        ), scheduledExecutorService, futureList, requestListener);



        if (fromDex) {
            Gson gson = new Gson();
            ContextActionConfigBean config = gson.fromJson(
                    FileUtils.getFileContent(SAVE_PATH + "config.json"),
                    ContextActionConfigBean.class
            );

            if (config != null) {
                // firstly schedule timed behavior, because it may use log to record contexts and actions

                for (ContextActionConfigBean.ContextConfigBean bean: config.getContext()) {
                    if (bean == null) {
                        continue;
                    }
                    ContextConfig contextConfig = new ContextConfig();
                    contextConfig.setContext(bean.getBuiltInContext());
                    contextConfig.setSensorType(bean.getSensorType().stream().map(SensorType::fromString).collect(Collectors.toList()));
                    for (int i = 0; i < bean.getIntegerParamKey().size(); i++) {
                        contextConfig.putValue(bean.getIntegerParamKey().get(i), bean.getIntegerParamValue().get(i));
                    }
                    for (int i = 0; i < bean.getLongParamKey().size(); i++) {
                        contextConfig.putValue(bean.getLongParamKey().get(i), bean.getLongParamValue().get(i));
                    }
                    for (int i = 0; i < bean.getFloatParamKey().size(); i++) {
                        contextConfig.putValue(bean.getFloatParamKey().get(i), bean.getFloatParamValue().get(i));
                    }
                    for (int i = 0; i < bean.getBooleanParamKey().size(); i++) {
                        contextConfig.putValue(bean.getBooleanParamKey().get(i), bean.getBooleanParamValue().get(i));
                    }
                    switch (contextConfig.getContext()) {
                        case "Proximity":
                            ProximityContext proximityContext = new ProximityContext(mContext, contextConfig, requestListener, Arrays.asList(this, contextListener), scheduledExecutorService, futureList);
                            contexts.add(proximityContext);
                            break;
                        case "Table":
                            TableContext tableContext = new TableContext(mContext, contextConfig, requestListener, Arrays.asList(this, contextListener), scheduledExecutorService, futureList);
                            contexts.add(tableContext);
                            break;
                        default:
                            break;
                    }
                }

                for (ContextActionConfigBean.ActionConfigBean bean: config.getAction()) {
                    if (bean == null) {
                        continue;
                    }
                    ActionConfig actionConfig = new ActionConfig();
                    actionConfig.setAction(bean.getBuiltInAction());
                    actionConfig.setSensorType(bean.getSensorType().stream().map(SensorType::fromString).collect(Collectors.toList()));
                    for (int i = 0; i < bean.getIntegerParamKey().size(); i++) {
                        actionConfig.putValue(bean.getIntegerParamKey().get(i), bean.getIntegerParamValue().get(i));
                    }
                    for (int i = 0; i < bean.getLongParamKey().size(); i++) {
                        actionConfig.putValue(bean.getLongParamKey().get(i), bean.getLongParamValue().get(i));
                    }
                    for (int i = 0; i < bean.getFloatParamKey().size(); i++) {
                        actionConfig.putValue(bean.getFloatParamKey().get(i), bean.getFloatParamValue().get(i));
                    }
                    for (int i = 0; i < bean.getBooleanParamKey().size(); i++) {
                        actionConfig.putValue(bean.getBooleanParamKey().get(i), bean.getBooleanParamValue().get(i));
                    }
                    switch (actionConfig.getAction()) {
                        case "TapTap":
                            TapTapAction tapTapAction = new TapTapAction(mContext, actionConfig, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList);
                            actions.add(tapTapAction);
                            break;
                        case "TopTap":
                            TopTapAction topTapAction = new TopTapAction(mContext, actionConfig, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList);
                            actions.add(topTapAction);
                            break;
                        case "Flip":
                            LogCollector flipLogCollector = collectorManager.newLogCollector("Flip", 800);
                            FlipAction flipAction = new FlipAction(mContext, actionConfig, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList, null);
                            actions.add(flipAction);
                            break;
                        case "Close":
                            CloseAction closeAction = new CloseAction(mContext, actionConfig, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList, null);
                            actions.add(closeAction);
                            break;
                        case "Pocket":
                            PocketAction pocketAction = new PocketAction(mContext, actionConfig, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList);
                            actions.add(pocketAction);
                            break;
                        case "Motion":
                            MotionAction motionAction = new MotionAction(mContext, actionConfig, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList);
                            actions.add(motionAction);
                            break;
                        default:
                            break;
                    }
                }
            }
        }

        this.dataDistributor = new DataDistributor(actions, contexts, contextLock);
        collectorManager.registerListener(dataDistributor);
    }

    private void setLogCollector(Class c, LogCollector logCollector) {
        for (BaseContext context: contexts) {
            if (c.isInstance(context)) {
                context.setLogCollector(logCollector);
            }
        }

        for (BaseAction action: actions) {
            if (c.isInstance(action)) {
                action.setLogCollector(logCollector);
            }
        }
    }

    public void startCollectors() {
        try {
            contextLock.lock();

            uploader = new Uploader(mContext, scheduledExecutorService, futureList, requestListener, handler);
            clickTrigger = new ClickTrigger(mContext, collectorManager, scheduledExecutorService, futureList);

            Gson gson = new Gson();
            ContextActionConfigBean config = gson.fromJson(
                    FileUtils.getFileContent(SAVE_PATH + "config.json"),
                    ContextActionConfigBean.class
            );

            // cwh: do not use Arrays.asList() to assign to collectors,
            // because it returns a fixed-size list backed by the specified array and we cannot perform add()
            collectors = new ArrayList<>();
            TimedCollector timedCollector = new TimedCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader);
            collectors.add(timedCollector);
            collectors.add(new TapTapCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader));
            collectors.add(new ConfigCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader));


            if (config != null) {
                // firstly schedule timed behavior, because it may use log to record contexts and actions
                if (config.getTimed() != null) {
                    for (ContextActionConfigBean.TimedConfigBean bean : config.getTimed()) {
                        if (bean == null) {
                            continue;
                        }
                        if (bean.getName() == null) {
                            continue;
                        }
                        if ("TriggerLog".equals(bean.getBuiltInSensor())) {
                            // Trigger log records all trigger events
                            LogCollector triggerLogCollector = collectorManager.newLogCollector("Trigger", 8192);
                            clickTrigger.setTriggerLogCollector(triggerLogCollector);
                            timedCollector.scheduleTimedLogUpload(triggerLogCollector, bean.getPeriodOrDelay(), bean.getInitialDelay(), bean.getName());
                        } else if ("ContextActionLog".equals(bean.getBuiltInSensor())) {
                            // ContextAction log records all contexts and actions
                            this.contextActionLogCollector = collectorManager.newLogCollector("ContextAction", 8192);
                            timedCollector.scheduleTimedLogUpload(contextActionLogCollector, bean.getPeriodOrDelay(), bean.getInitialDelay(), bean.getName());
                        } else {
                            CollectorManager.CollectorType type;
                            try {
                                type = CollectorManager.CollectorType.valueOf(bean.getBuiltInSensor());
                            } catch (Exception e) {
                                continue;
                            }
                            if (type != CollectorManager.CollectorType.Log) {
                                TriggerConfig triggerConfig = bean.getTriggerConfig();
                                if (triggerConfig == null) {
                                    triggerConfig = new TriggerConfig();
                                }
                                if (bean.isFixedDelay()) {
                                    timedCollector.scheduleFixedDelayUpload(type, triggerConfig, bean.getPeriodOrDelay(), bean.getInitialDelay(), bean.getName());
                                } else {
                                    timedCollector.scheduleFixedRateUpload(type, triggerConfig, bean.getPeriodOrDelay(), bean.getInitialDelay(), bean.getName());
                                }
                            }
                        }
                    }
                }

                if (config.getContext() != null) {
                    for (ContextActionConfigBean.ContextConfigBean bean : config.getContext()) {
                        if (bean == null) {
                            continue;
                        }
                        if (bean.getBuiltInContext() == null) {
                            continue;
                        }
                        ContextConfig contextConfig = new ContextConfig();
                        contextConfig.setContext(bean.getBuiltInContext());
                        contextConfig.setSensorType(bean.getSensorType().stream().map(SensorType::fromString).collect(Collectors.toList()));
                        for (int i = 0; i < bean.getIntegerParamKey().size(); i++) {
                            contextConfig.putValue(bean.getIntegerParamKey().get(i), bean.getIntegerParamValue().get(i));
                        }
                        for (int i = 0; i < bean.getLongParamKey().size(); i++) {
                            contextConfig.putValue(bean.getLongParamKey().get(i), bean.getLongParamValue().get(i));
                        }
                        for (int i = 0; i < bean.getFloatParamKey().size(); i++) {
                            contextConfig.putValue(bean.getFloatParamKey().get(i), bean.getFloatParamValue().get(i));
                        }
                        for (int i = 0; i < bean.getBooleanParamKey().size(); i++) {
                            contextConfig.putValue(bean.getBooleanParamKey().get(i), bean.getBooleanParamValue().get(i));
                        }
                        Number initialDelay = contextConfig.getValue("intialDelay");
                        Number period = contextConfig.getValue("period");
                        String name = contextConfig.getString("name");
                        switch (contextConfig.getContext()) {
                            case "Informational":
                                LogCollector informationLogCollector = collectorManager.newLogCollector("Informational", 8192);
                                contexts.add(new InformationalContext(mContext, contextConfig, requestListener, Arrays.asList(this, contextListener), informationLogCollector, scheduledExecutorService, futureList));
                                //                            setLogCollector(InformationalContext.class, informationLogCollector);
                                collectors.add(new InformationalContextCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader, informationLogCollector));
                                timedCollector.scheduleTimedLogUpload(
                                        informationLogCollector,
                                        (period == null) ? 30 * 60000 : period.longValue(),
                                        (initialDelay == null) ? 60000 : initialDelay.longValue(),
                                        (name == null) ? "Informational" : name
                                );
                                break;
                            case "Config":
                                LogCollector configLogCollector = collectorManager.newLogCollector("Config", 8192);
                                contexts.add(new ConfigContext(mContext, contextConfig, requestListener, Arrays.asList(this, contextListener), configLogCollector, scheduledExecutorService, futureList));
                                //                            setLogCollector(ConfigContext.class, configLogCollector);
                                timedCollector.scheduleTimedLogUpload(
                                        configLogCollector,
                                        (period == null) ? 30 * 60000 : period.longValue(),
                                        (initialDelay == null) ? 5000 : initialDelay.longValue(),
                                        (name == null) ? "Config" : name
                                );
                                break;
                            default:
                                break;
                        }
                    }
                }

                if (config.getAction() != null) {
                    for (ContextActionConfigBean.ActionConfigBean bean : config.getAction()) {
                        if (bean == null) {
                            continue;
                        }
                        if (bean.getBuiltInAction() == null) {
                            continue;
                        }
                        ActionConfig actionConfig = new ActionConfig();
                        actionConfig.setAction(bean.getBuiltInAction());
                        actionConfig.setSensorType(bean.getSensorType().stream().map(SensorType::fromString).collect(Collectors.toList()));
                        for (int i = 0; i < bean.getIntegerParamKey().size(); i++) {
                            actionConfig.putValue(bean.getIntegerParamKey().get(i), bean.getIntegerParamValue().get(i));
                        }
                        for (int i = 0; i < bean.getLongParamKey().size(); i++) {
                            actionConfig.putValue(bean.getLongParamKey().get(i), bean.getLongParamValue().get(i));
                        }
                        for (int i = 0; i < bean.getFloatParamKey().size(); i++) {
                            actionConfig.putValue(bean.getFloatParamKey().get(i), bean.getFloatParamValue().get(i));
                        }
                        for (int i = 0; i < bean.getBooleanParamKey().size(); i++) {
                            actionConfig.putValue(bean.getBooleanParamKey().get(i), bean.getBooleanParamValue().get(i));
                        }
                        switch (actionConfig.getAction()) {
                            case "Flip":
                                Log.e("upload:", "register Flip LogCollector");
                                LogCollector flipLogCollector = collectorManager.newLogCollector("Flip", 800);
                                collectors.add(new FlipCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader, flipLogCollector));
                                setLogCollector(FlipAction.class, flipLogCollector);
                                break;
                            case "Close":
                                Log.e("upload:", "register Close LogCollector");
                                LogCollector closeLogCollector = collectorManager.newLogCollector("Close", 800);
                                collectors.add(new CloseCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader, closeLogCollector));
                                setLogCollector(CloseAction.class, closeLogCollector);
                                break;
                            default:
                                break;
                        }
                    }
                }

                // register broadcast receiver
                IntentFilter intentFilter = new IntentFilter();
                if (config.getListenedSystemActions() != null) {
                    config.getListenedSystemActions().stream().filter(Objects::nonNull).forEach(intentFilter::addAction);
                }
                if (!config.isOverrideSystemActions()) {
                    Arrays.stream(listenedActions).filter(Objects::nonNull).forEach(intentFilter::addAction);
                }
                mContext.registerReceiver(mBroadcastReceiver, intentFilter, null, handler);
                Log.e("OverrideSystemActions", Boolean.toString(config.isOverrideSystemActions()));
                intentFilter.actionsIterator().forEachRemaining(item -> Log.e("Register broadcast", item));

                // register content observer
                if (config.getListenedSystemURIs() != null) {
                    config.getListenedSystemURIs().stream().filter(Objects::nonNull).map(Uri::parse).forEach(this::registerURI);
                }
                if (!config.isOverrideSystemURIs()) {
                    Arrays.stream(listenedURIs).forEach(this::registerURI);
                }
                Log.e("OverrideSystemURIs", Boolean.toString(config.isOverrideSystemURIs()));
                mRegURIs.forEach(uri -> Log.e("Register URI", uri.toString()));
            }
        } finally {
            contextLock.unlock();
        }
    }

    private void monitorAction() {
        actionFuture = scheduledExecutorService.scheduleAtFixedRate(() -> {
            try {
                for (BaseAction action : actions) {
                    action.getAction();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, 3000L, 20L, TimeUnit.MILLISECONDS);
        futureList.add(actionFuture);
    }

    private void monitorContext() {
        contextFuture = scheduledExecutorService.scheduleAtFixedRate(() -> {
            try {
                for (BaseContext context: contexts) {
                    context.getContext();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, 3000L, 1000L, TimeUnit.MILLISECONDS);
        futureList.add(contextFuture);
    }

    public void onSensorChangedDex(SensorEvent event) {
    }

    public void onAccessibilityEventDex(AccessibilityEvent event) {
        Heart.getInstance().newSensorGetEvent("Accessibility", System.currentTimeMillis());
        if (handler != null) {
            final AccessibilityEvent event1 = AccessibilityEvent.obtain(event);
            handler.post(() -> {
                if (dataDistributor != null) {
                    dataDistributor.onAccessibilityEvent(event1);
                }
                event1.recycle();
            });
        }
    }

    public void onKeyEventDex(KeyEvent event) {
        BroadcastEvent bc_event = new BroadcastEvent(
                System.currentTimeMillis(),
                "KeyEvent",
                "KeyEvent://"+event.getAction()+"/"+event.getKeyCode()
        );
        bc_event.getExtras().putInt("action", event.getAction());
        bc_event.getExtras().putInt("code", event.getKeyCode());
        bc_event.getExtras().putInt("source", event.getSource());
        bc_event.getExtras().putLong("eventTime", event.getEventTime());
        bc_event.getExtras().putLong("downTime", event.getDownTime());
        onBroadcastEventDex(bc_event);
    }

    public void onBroadcastEventDex(BroadcastEvent event) {
        Heart.getInstance().newSensorGetEvent("Broadcast", System.currentTimeMillis());
        if (handler != null) {
            handler.post(() -> {
                if (dataDistributor != null) {
                    dataDistributor.onBroadcastEvent(event);
                }
            });
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onAction(ActionResult action) {
        Heart.getInstance().newActionTriggerEvent(action.getAction(), action.getTimestamp());
        assignIDAndRecord(action);
        if (collectors != null) {
            for (BaseCollector collector: collectors) {
                collector.onAction(action);
            }
        }
    }

    @Override
    public void onContext(ContextResult context) {
        Heart.getInstance().newContextTriggerEvent(context.getContext(), context.getTimestamp());
        assignIDAndRecord(context);
        if (collectors != null) {
            for (BaseCollector collector: collectors) {
                collector.onContext(context);
            }
        }
    }

    class CustomBroadcastReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            BroadcastEvent event = new BroadcastEvent(
                    System.currentTimeMillis(),
                    "BroadcastReceive",
                    intent.getAction()
            );
            event.setExtras(intent.getExtras());
            onBroadcastEventDex(event);
        }
    }

    class CustomContentObserver extends ContentObserver {
        public CustomContentObserver(Handler handler) {
            super(handler);
        }

        @Override
        public void onChange(boolean selfChange) {
            onChange(selfChange, null);
        }

        @Override
        public void onChange(boolean selfChange, @Nullable Uri uri) {
            BroadcastEvent event = new BroadcastEvent(
                    System.currentTimeMillis(),
                    "ContentChange",
                    (uri == null)? "uri_null" : uri.toString()
            );
            onBroadcastEventDex(event);
        }
    }

    private void registerURI(Uri uri) {
        if (uri != null) {
            if (!mRegURIs.contains(uri)) {
                mContext.getContentResolver().registerContentObserver(uri, true, mContentObserver);
                mRegURIs.add(uri);
            }
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private int incContextActionID() {
        return mContextActionIDCounter.getAndIncrement();
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void assignIDAndRecord(Result contextOrAction) {
        int contextActionID = incContextActionID();
        /*
            Logging:
                timestamp | contextActionID | contextOrAction | reason | extras

         */
        if (contextActionLogCollector != null) {
            String line = contextOrAction.getTimestamp() + "\t" +
                    contextActionID + "\t" +
                    contextOrAction.getKey() + "\t" +
                    contextOrAction.getReason() + "\t" +
                    JSONUtils.bundleToJSON(contextOrAction.getExtras()).toString();
            contextActionLogCollector.addLog(line);
        }
        // assign ID
        contextOrAction.getExtras().putInt("contextActionID", contextActionID);
    }

    public Heartbeat getHeartbeat() {
        int size = 0;
        int alive = 0;
        synchronized (futureList) {
            size = futureList.size();
            for (int i = 0; i < size; i++) {
                if (!futureList.get(i).isCancelled() && !futureList.get(i).isDone()) {
                    alive++;
                }
            }
        }
        Heart.getInstance().setFutureCount(size);
        Heart.getInstance().setAliveFutureCount(alive);
        return Heart.getInstance().beat();
    }
}
