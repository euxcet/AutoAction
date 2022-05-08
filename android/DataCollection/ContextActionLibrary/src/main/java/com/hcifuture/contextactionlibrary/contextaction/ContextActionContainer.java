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
import android.provider.Settings;
import android.util.Log;
import android.view.KeyEvent;
import android.view.accessibility.AccessibilityEvent;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;

import com.google.gson.Gson;
import com.hcifuture.contextactionlibrary.contextaction.action.ExampleAction;
import com.hcifuture.contextactionlibrary.contextaction.action.MotionAction;
import com.hcifuture.contextactionlibrary.contextaction.collect.CloseCollector;
import com.hcifuture.contextactionlibrary.contextaction.collect.ConfigCollector;
import com.hcifuture.contextactionlibrary.contextaction.collect.ExampleCollector;
import com.hcifuture.contextactionlibrary.contextaction.collect.FlipCollector;
import com.hcifuture.contextactionlibrary.contextaction.collect.InformationalContextCollector;
import com.hcifuture.contextactionlibrary.contextaction.collect.TapTapCollector;
import com.hcifuture.contextactionlibrary.contextaction.collect.TimedCollector;
import com.hcifuture.contextactionlibrary.contextaction.context.ConfigContext;
import com.hcifuture.contextactionlibrary.contextaction.context.informational.InformationalContext;
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
import com.hcifuture.contextactionlibrary.utils.FileUtils;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.contextactionlibrary.contextaction.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.SensorType;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
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

    private ScheduledFuture<?> actionFuture;
    private ScheduledFuture<?> contextFuture;

    private CollectorManager collectorManager;
    private DataDistributor dataDistributor;

    private static String SAVE_PATH;

    private final CustomBroadcastReceiver mBroadcastReceiver;
    private final CustomContentObserver mContentObserver;
    private final List<Uri> mRegURIs;

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

        // clickTrigger = new ClickTrigger(context, Arrays.asList(Trigger.CollectorType.CompleteIMU, Trigger.CollectorType.Bluetooth));
        this.futureList = new ArrayList<>();

        // scheduleCleanData();

        mBroadcastReceiver = new CustomBroadcastReceiver();
        mContentObserver = new CustomContentObserver(new Handler());
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
        /*
        if (clickTrigger != null) {
            clickTrigger.resume();
        }
         */
    }

    public void stop() {
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

        /*
        if (openSensor) {
            for (BaseSensorManager sensorManager: sensorManagers) {
                sensorManager.stop();
            }
        }
         */
        for (BaseAction action: actions) {
            action.stop();
        }
        for (BaseContext context: contexts) {
            context.stop();
        }
        /*
        if (clickTrigger != null) {
            clickTrigger.pause();
            clickTrigger.close();
        }
         */
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
        for (ScheduledFuture<?> future: futureList) {
            future.cancel(true);
        }
        futureList.clear();
        scheduledExecutorService.shutdownNow();
    }

    public void pause() {
        for (BaseAction action: actions) {
            action.stop();
        }
        for (BaseContext context: contexts) {
            context.stop();
        }
        if (collectorManager != null) {
            collectorManager.pause();
        }
        /*
        if (clickTrigger != null) {
            clickTrigger.pause();
        }
         */
    }

    public void resume() {
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
        /*
        if (clickTrigger != null) {
            clickTrigger.resume();
        }
         */
        if (collectorManager != null) {
            collectorManager.resume();
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

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void initialize() {
        this.scheduledExecutorService = Executors.newScheduledThreadPool(32);
        ((ScheduledThreadPoolExecutor)scheduledExecutorService).setRemoveOnCancelPolicy(true);

        this.collectorManager = new CollectorManager(mContext, Arrays.asList(
                CollectorManager.CollectorType.IMU,
                CollectorManager.CollectorType.Location,
                CollectorManager.CollectorType.Audio,
                CollectorManager.CollectorType.Bluetooth,
                CollectorManager.CollectorType.Wifi,
                CollectorManager.CollectorType.GPS,
                CollectorManager.CollectorType.NonIMU
        ), scheduledExecutorService, futureList);

        this.clickTrigger = new ClickTrigger(mContext, collectorManager, scheduledExecutorService, futureList);
        this.uploader = new Uploader(mContext, scheduledExecutorService, futureList);

        // cwh: do not use Arrays.asList() to assign to collectors,
        // because it returns a fixed-size list backed by the specified array and we cannot perform add()
        collectors = new ArrayList<>();
        TimedCollector timedCollector = new TimedCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader);
        collectors.add(timedCollector);
        collectors.add(new TapTapCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader));
        ConfigCollector configCollector = new ConfigCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader);
        collectors.add(configCollector);

        if (fromDex) {
            Gson gson = new Gson();
            ContextActionConfigBean config = gson.fromJson(
                    FileUtils.getFileContent(SAVE_PATH + "config.json"),
                    ContextActionConfigBean.class
            );

            if (config != null) {
                // firstly schedule timed behavior, because it may use log to record contexts and actions
                for (ContextActionConfigBean.TimedConfigBean bean: config.getTimed()) {
                    if (bean == null) {
                        continue;
                    }
                    CollectorManager.CollectorType type;
                    try {
                        type = CollectorManager.CollectorType.valueOf(bean.getBuiltInSensor());
                    } catch (Exception e) {
                        continue;
                    }
                    if (bean.getName() == null) {
                        continue;
                    }
                    // ContextAction log records all contexts and actions
                    if (type == CollectorManager.CollectorType.Log) {
                        LogCollector contextActionLogCollector = collectorManager.newLogCollector("ContextAction", 8192);
                        configCollector.setContextActionLogCollector(contextActionLogCollector);
                        timedCollector.scheduleTimedLogUpload(contextActionLogCollector, bean.getPeriodOrDelay(), bean.getInitialDelay(), bean.getName());
                        Log.e("TimedCollector", "register fixed rate upload log: ContextAction" +
                                " delay: " + bean.getPeriodOrDelay() +
                                " initialDelay: " + bean.getInitialDelay() +
                                " name: " + bean.getName());
                    } else {
                        TriggerConfig triggerConfig = bean.getTriggerConfig();
                        if (triggerConfig == null) {
                            triggerConfig = new TriggerConfig();
                        }
                        if (bean.isFixedDelay()) {
                            timedCollector.scheduleFixedDelayUpload(type, triggerConfig, bean.getPeriodOrDelay(), bean.getInitialDelay(), bean.getName());
                            Log.e("TimedCollector", "register fixed delay upload: " + type.name() +
                                    " delay: " + bean.getPeriodOrDelay() +
                                    " initialDelay: " + bean.getInitialDelay() +
                                    " name: " + bean.getName());
                        } else {
                            timedCollector.scheduleFixedRateUpload(type, triggerConfig, bean.getPeriodOrDelay(), bean.getInitialDelay(), bean.getName());
                            Log.e("TimedCollector", "register fixed rate upload: " + type.name() +
                                    " period: " + bean.getPeriodOrDelay() +
                                    " initialDelay: " + bean.getInitialDelay() +
                                    " name: " + bean.getName());
                        }
                    }
                }

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
                        case "Informational":
                            LogCollector informationLogCollector = collectorManager.newLogCollector("Informational", 8192);
                            timedCollector.scheduleTimedLogUpload(informationLogCollector, 60000, 5000, "Informational");
                            collectors.add(new InformationalContextCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader, informationLogCollector));
                            InformationalContext informationalContext = new InformationalContext(mContext, contextConfig, requestListener, Arrays.asList(this, contextListener),informationLogCollector, scheduledExecutorService, futureList);
                            contexts.add(informationalContext);
                            break;
                        case "Config":
                            LogCollector configLogCollector = collectorManager.newLogCollector("Config", 8192);
                            Number initialDelay = contextConfig.getValue("intialDelay");
                            Number period = contextConfig.getValue("period");
                            String name = contextConfig.getString("name");
                            initialDelay = (initialDelay == null)? 5000 : initialDelay;
                            period = (period == null)? 60000 : period;
                            name = (name == null)? "Config" : name;
                            timedCollector.scheduleTimedLogUpload(configLogCollector, period.longValue(), initialDelay.longValue(), name);
                            ConfigContext configContext = new ConfigContext(mContext, contextConfig, requestListener, Arrays.asList(this, contextListener), configLogCollector, scheduledExecutorService, futureList);
                            contexts.add(configContext);
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
                            LogCollector FliplogCollector = collectorManager.newLogCollector("Flip", 800);
                            collectors.add(new FlipCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader, FliplogCollector));
                            FlipAction flipAction = new FlipAction(mContext, actionConfig, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList, FliplogCollector);
                            actions.add(flipAction);
                            break;
                        case "Close":
                            LogCollector CloselogCollector = collectorManager.newLogCollector("Close", 800);
                            collectors.add(new CloseCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader, CloselogCollector));
                            CloseAction closeAction = new CloseAction(mContext, actionConfig, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList, CloselogCollector);
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
                        case "Example":
                            LogCollector logCollector = collectorManager.newLogCollector("Log0", 100);
                            timedCollector.scheduleTimedLogUpload(logCollector, 5000, 0, "Example");
                            collectors.add(new ExampleCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader, logCollector));
                            ExampleAction exampleAction = new ExampleAction(mContext, actionConfig, requestListener, Arrays.asList(this, actionListener), logCollector, scheduledExecutorService, futureList);
                            actions.add(exampleAction);
                            break;
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
                mContext.registerReceiver(mBroadcastReceiver, intentFilter);
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
        }


            /*
            for (int i = 0; i < actionConfig.size(); i++) {
                ActionConfig config = actionConfig.get(i);
                switch (config.getAction()) {
                    case "TapTap":
                        tapTapAction = new TapTapAction(mContext, config, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList);
                        actions.add(tapTapAction);
                        break;
                    case "TopTap":
                        TopTapAction topTapAction = new TopTapAction(mContext, config, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList);
                        actions.add(topTapAction);
                        break;
                    case "Flip":
                        LogCollector FliplogCollector = collectorManager.newLogCollector("Flip", 800);
                        collectors.add(new FlipCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, FliplogCollector));
                        FlipAction flipAction = new FlipAction(mContext, config, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList, FliplogCollector);
                        actions.add(flipAction);
                        break;
                    case "Close":
                        CloseAction closeAction = new CloseAction(mContext, config, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList);
                        actions.add(closeAction);
                        break;
                    case "Pocket":
                        PocketAction pocketAction = new PocketAction(mContext, config, requestListener, Arrays.asList(this, actionListener), scheduledExecutorService, futureList);
                        actions.add(pocketAction);
                        break;
                    case "Example":
                        LogCollector logCollector = collectorManager.newLogCollector("Log0", 100);
                        timedCollector.scheduleTimedLogUpload(logCollector, 5000, 0, "Example");
                        collectors.add(new ExampleCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, logCollector));
                        ExampleAction exampleAction = new ExampleAction(mContext, config, requestListener, Arrays.asList(this, actionListener), logCollector, scheduledExecutorService, futureList);
                        actions.add(exampleAction);
                        break;
                }
            }
            for (int i = 0; i < contextConfig.size(); i++) {
                ContextConfig config = contextConfig.get(i);
                switch (config.getContext()) {
                    case "Proximity":
                        ProximityContext proximityContext = new ProximityContext(mContext, config, requestListener, Arrays.asList(this, contextListener), scheduledExecutorService, futureList);
                        contexts.add(proximityContext);
                        break;
                    case "Table":
                        TableContext tableContext = new TableContext(mContext, config, requestListener, Arrays.asList(this, contextListener), scheduledExecutorService, futureList);
                        contexts.add(tableContext);
                        break;
                    case "Informational":
                        LogCollector informationLogCollector = collectorManager.newLogCollector("Informational", 8192);
                        timedCollector.scheduleTimedLogUpload(informationLogCollector, 60000, 5000, "Informational");
                        collectors.add(new InformationalContextCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, informationLogCollector));
                        InformationalContext informationalContext = new InformationalContext(mContext, config, requestListener, Arrays.asList(this, contextListener),informationLogCollector, scheduledExecutorService, futureList);
                        contexts.add(informationalContext);
                        break;
                    case "Config":
                        LogCollector configLogCollector = collectorManager.newLogCollector("Config", 8192);
                        timedCollector.scheduleTimedLogUpload(configLogCollector, 60000, 5000, "Config");
                        collectors.add(new ConfigCollector(mContext, scheduledExecutorService, futureList, requestListener, clickTrigger, configLogCollector));
                        ConfigContext configContext = new ConfigContext(mContext, config, requestListener, Arrays.asList(this, contextListener), configLogCollector, scheduledExecutorService, futureList);
                        contexts.add(configContext);
                        break;
                }
            }

             */
        this.dataDistributor = new DataDistributor(actions, contexts);
        collectorManager.registerListener(dataDistributor);


        // init sensor
        /*
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
         */
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
        int type = event.sensor.getType();
        /*
        for (BaseSensorManager sensorManager: sensorManagers) {
            if (sensorManager.getSensorTypeList() != null && sensorManager.getSensorTypeList().contains(type)) {
                sensorManager.onSensorChangedDex(event);
            }
        }
         */
    }

    public void onAccessibilityEventDex(AccessibilityEvent event) {
        for (BaseContext context: contexts) {
            context.onAccessibilityEvent(event);
        }
        /*
        for (BaseSensorManager sensorManager: sensorManagers) {
            sensorManager.onAccessibilityEventDex(event);
        }
         */
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
        for (BaseContext context: contexts) {
            context.onBroadcastEvent(event);
        }
    }

    /*
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
                if (clickTrigger != null) {
                    clickTrigger.cleanData();
                }
                Log.e("TapTapCollector", "triggered");
            }
        }, date, 24 * 60 * 60 * 1000);

    }
     */

    /*
    @Override
    public void onActionRecognized(ActionResult action) {
        if (action.getAction().equals("TapTapConfirmed")) {
            markTimestamp = action.getTimestamp();
            tapTapAction.onConfirmed();
        }
        else if (action.getAction().equals("TopTap")) {
            markTimestamp = action.getTimestamp();
        }
    }
     */

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onAction(ActionResult action) {
        if (collectors != null) {
            for (BaseCollector collector: collectors) {
                collector.onAction(action);
            }
        }
            /*
        if (action.getAction().equals("TapTap") || action.getAction().equals("TopTap") || action.getAction().equals("Pocket")) {
            if (clickTrigger != null) {
                clickTrigger.trigger(Collections.singletonList(Trigger.CollectorType.CompleteIMU), new TriggerConfig());
            }
        }
             */
//        if (collectors != null) {
//            for (BaseCollector collector: collectors) {
//                collector.onAction(action);
//            }
//        }
    }

    /*
    @Override
    public void onActionSave(ActionResult action) {
        if (action.getAction().equals("TapTap") || action.getAction().equals("TopTap")) {
            action.setTimestamp(markTimestamp);
        }
        if (collectors != null) {
            for (BaseCollector collector : collectors) {
                collector.onAction(action);
            }
        }
    }
     */

    @Override
    public void onContext(ContextResult context) {
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

    void registerURI(Uri uri) {
        if (uri != null) {
            if (!mRegURIs.contains(uri)) {
                mContext.getContentResolver().registerContentObserver(uri, true, mContentObserver);
                mRegURIs.add(uri);
            }
        }
    }
}
