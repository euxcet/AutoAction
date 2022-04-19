package com.hcifuture.datacollection.contextaction;

import static android.content.Context.MODE_PRIVATE;

import android.accessibilityservice.AccessibilityService;
import android.bluetooth.BluetoothDevice;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.database.ContentObserver;
import android.net.Uri;
import android.net.wifi.WifiManager;
import android.os.Handler;
import android.provider.Settings;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;

import com.google.gson.Gson;
import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.shared.communicate.SensorType;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.shared.communicate.config.RequestConfig;
import com.hcifuture.shared.communicate.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.RequestResult;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

import androidx.annotation.Nullable;
import dalvik.system.DexClassLoader;

public class LoaderManager {
    private AccessibilityService mService;
    private DexClassLoader classLoader;
    private ContextActionLoader loader;
    private final List<String> UPDATABLE_FILES = Arrays.asList(
            "param_dicts.json",
            "pages.csv",
            "tasks.csv",
            "param_max.json",
            "words.csv",
            "classes.dex",
            "release.dex",
            "config.json",
            "tap7cls_pixel4.tflite",
            "ResultModel.tflite",
            "combined.tflite",
            "pocket.tflite"
    );
    private AtomicBoolean isUpgrading;
    private ContextListener contextListener;
    private ActionListener actionListener;

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
            // WiFi related
            WifiManager.NETWORK_STATE_CHANGED_ACTION,
            WifiManager.WIFI_STATE_CHANGED_ACTION
    };

    private final CustomBroadcastReceiver mBroadcastReceiver;
    private final CustomContentObserver mContentObserver;
    private final List<Uri> mRegURIs;

    public LoaderManager(AccessibilityService service, ContextListener contextListener, ActionListener actionListener) {
        this.mService = service;
        this.isUpgrading = new AtomicBoolean(false);
        if (contextListener == null) {
            this.contextListener = x -> {};
        } else {
            this.contextListener = contextListener;
        }
        if (actionListener == null) {
            this.actionListener = x -> {};
        } else {
            this.actionListener = actionListener;
        }
        mBroadcastReceiver = new CustomBroadcastReceiver();
        mContentObserver = new CustomContentObserver(new Handler());
        mRegURIs = new ArrayList<>();
        calculateLocalMD5(UPDATABLE_FILES);
    }

    private RequestResult handleRequest(RequestConfig config) {
        RequestResult result = new RequestResult();
        if (config.getString("getAppTapBlockValue") != null) {
            result.putValue("getAppTapBlockValueResult", 0);
        }

        if (config.getBoolean("getCanTapTap") != null) {
            result.putValue("getCanTapTapResult", true);
        }

        if (config.getBoolean("getCanTopTap") != null) {
            result.putValue("getCanTopTapResult", true);
        }

        if (config.getValue("getWindows") != null) {
            result.putObject("getWindows", mService.getWindows());
        }

        return result;
    }

    private void calculateLocalMD5(List<String> filenames) {
        SharedPreferences fileMD5 = mService.getSharedPreferences("FILE_MD5", MODE_PRIVATE);
        SharedPreferences.Editor editor = fileMD5.edit();
        for (String filename: filenames) {
            editor.putString(filename, FileUtils.fileToMD5(BuildConfig.SAVE_PATH + filename));
        }
        editor.apply();
    }

    private void updateLocalMD5(List<String> changedFilename, List<String> serverMD5s) {
        SharedPreferences fileMD5 = mService.getSharedPreferences("FILE_MD5", MODE_PRIVATE);
        SharedPreferences.Editor editor = fileMD5.edit();
        for (int i = 0; i < changedFilename.size(); i++) {
            editor.putString(changedFilename.get(i), serverMD5s.get(i));
        }
        editor.apply();
    }

    public void start() {
        FileUtils.checkFiles(mService, UPDATABLE_FILES, (changedFilename, serverMD5s) -> {
            FileUtils.downloadFiles(mService, changedFilename, () -> {
                updateLocalMD5(changedFilename, serverMD5s);
                loadContextActionLibrary();
            });
        });
    }

    public void upgrade() {
        if (isUpgrading.get()) {
            return;
        }
        calculateLocalMD5(UPDATABLE_FILES);
        FileUtils.checkFiles(mService, UPDATABLE_FILES, (changedFilename, serverMD5) -> {
            if (changedFilename.isEmpty()) {
                return;
            }
            isUpgrading.set(true);
            stop();
            new Timer().schedule(new TimerTask() {
                @Override
                public void run() {
                    FileUtils.downloadFiles(mService, changedFilename, () -> {
                        updateLocalMD5(changedFilename, serverMD5);
                        loadContextActionLibrary();
                        isUpgrading.set(false);
                    });
                }
            }, 5000);
        });
    }

    public void stop() {
        // unregister broadcast receiver
        mService.unregisterReceiver(mBroadcastReceiver);
        // unregister content observer
        mService.getContentResolver().unregisterContentObserver(mContentObserver);
        mRegURIs.clear();

        if (loader != null) {
            loader.stopDetection();
            loader = null;
            classLoader = null;
        }
    }

    public void resume() {
        if (loader != null) {
            loader.startDetection();
        }
    }

    public void pause() {
        if (loader != null) {
            loader.stopDetection();
        }
    }

    private void loadContextActionLibrary() {
        final File tmpDir = mService.getDir("dex", 0);
        classLoader = new DexClassLoader(BuildConfig.SAVE_PATH + "classes.dex", tmpDir.getAbsolutePath(), null, this.getClass().getClassLoader());
        loader = new ContextActionLoader(mService, classLoader);

        Gson gson = new Gson();
        ContextActionConfigBean config = gson.fromJson(
                FileUtils.getFileContent(BuildConfig.SAVE_PATH + "config.json"),
                ContextActionConfigBean.class
        );

        List<ContextConfig> contextConfigs = new ArrayList<>();
        List<ActionConfig> actionConfigs = new ArrayList<>();


        for (ContextActionConfigBean.ContextConfigBean bean: config.getContext()) {
            if (bean == null)
                continue;
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
            contextConfigs.add(contextConfig);
        }

        for (ContextActionConfigBean.ActionConfigBean bean: config.getAction()) {
            if (bean == null)
                continue;
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
            actionConfigs.add(actionConfig);
        }

        RequestListener requestListener = this::handleRequest;

        // register broadcast receiver
        IntentFilter intentFilter = new IntentFilter();
        config.getListenedSystemActions().stream().filter(Objects::nonNull).forEach(intentFilter::addAction);
        if (!config.isOverrideSystemActions()) {
            Arrays.stream(listenedActions).filter(Objects::nonNull).forEach(intentFilter::addAction);
        }
        intentFilter.actionsIterator().forEachRemaining(item -> Log.e("Register broadcast", item));
        mService.registerReceiver(mBroadcastReceiver, intentFilter);

        // register content observer
        config.getListenedSystemURIs().stream().filter(Objects::nonNull).map(Uri::parse).forEach(this::registerURI);
        if (!config.isOverrideSystemURIs()) {
            Arrays.stream(listenedURIs).forEach(this::registerURI);
        }
        mRegURIs.forEach(uri -> Log.e("Register URI", uri.toString()));

        loader.startDetection(actionConfigs, actionListener, contextConfigs, contextListener, requestListener);
        /*
        NcnnInstance.init(mService,
                BuildConfig.SAVE_PATH + "best.param",
                BuildConfig.SAVE_PATH + "best.bin",
                4,
                128,
                6,
                1,
                2);
        NcnnInstance ncnnInstance = NcnnInstance.getInstance();
        float[] data = new float[128 * 6];
        Arrays.fill(data, 0.1f);
        Log.e("result", ncnnInstance.actionDetect(data) + " ");
         */
    }

    public void onAccessibilityEvent(AccessibilityEvent event) {
        if (loader != null) {
            loader.onAccessibilityEvent(event);
        }
    }

    public void onBroadcastEvent(BroadcastEvent event) {
        if (loader != null) {
            loader.onBroadcastEvent(event);
        }
    }

    class CustomBroadcastReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            BroadcastEvent event = new BroadcastEvent(
                    System.currentTimeMillis(),
                    intent.getAction(),
                    "",
                    "BroadcastReceive",
                    intent.getExtras()
            );
            onBroadcastEvent(event);
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
                    (uri == null)? "uri_null" : uri.toString(),
                    "",
                    "ContentChange"
            );
            onBroadcastEvent(event);
        }
    }

    void registerURI(Uri uri) {
        if (uri != null) {
            if (!mRegURIs.contains(uri)) {
                mService.getContentResolver().registerContentObserver(uri, true, mContentObserver);
                mRegURIs.add(uri);
            }
        }
    }
}
