package com.example.datacollection.ui;

import android.accessibilityservice.AccessibilityService;
import android.app.Notification;
import android.app.PendingIntent;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;
import android.widget.Button;
import android.widget.Toast;

import com.example.datacollection.BuildConfig;
import com.example.datacollection.contextaction.ContextActionLoader;
import com.example.datacollection.utils.FileUtils;
import com.example.datacollection.utils.NetworkUtils;
import com.example.ncnnlibrary.communicate.BuiltInActionEnum;
import com.example.ncnnlibrary.communicate.BuiltInContextEnum;
import com.example.ncnnlibrary.communicate.SensorType;
import com.example.ncnnlibrary.communicate.config.ActionConfig;
import com.example.ncnnlibrary.communicate.config.ContextConfig;
import com.example.ncnnlibrary.communicate.config.RequestConfig;
import com.example.ncnnlibrary.communicate.event.ButtonActionEvent;
import com.example.ncnnlibrary.communicate.listener.ActionListener;
import com.example.ncnnlibrary.communicate.listener.ContextListener;
import com.example.ncnnlibrary.communicate.listener.RequestListener;
import com.example.ncnnlibrary.communicate.result.RequestResult;
import com.gyf.cactus.Cactus;
import com.gyf.cactus.callback.CactusCallback;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.Arrays;

import dalvik.system.DexClassLoader;

public class MainService extends AccessibilityService {
    private Context mContext;
    private DexClassLoader classLoader;
    private ContextActionLoader loader;
    private Handler mHandler;

    private HomeWatcherReceiver mHomeKeyReceiver;

    public MainService() {
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        if (loader != null) {
            loader.onAccessibilityEvent(event);
        }
    }

    @Override
    public void onInterrupt() {

    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return super.onStartCommand(intent, flags, startId);
    }

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        mContext = this;
        mHandler = new Handler(Looper.getMainLooper());
        Log.e("TEST", "onServiceConnected");

        mHomeKeyReceiver = new HomeWatcherReceiver();
        final IntentFilter homeFilter = new IntentFilter(Intent.ACTION_CLOSE_SYSTEM_DIALOGS);
        homeFilter.addAction(Intent.ACTION_SCREEN_ON);
        homeFilter.addAction(Intent.ACTION_SCREEN_OFF);
        registerReceiver(mHomeKeyReceiver, homeFilter);

        String[] downloadFileNames = new String[]{"param_dicts.json", "param_max.json", "words.csv"};
        for (String fileName: downloadFileNames) {
            NetworkUtils.downloadFile(this, fileName, new FileCallback() {
                @Override
                public void onSuccess(Response<File> response) {
                    File file = response.body();
                    File saveFile = new File(BuildConfig.SAVE_PATH, fileName);
                    FileUtils.copy(file, saveFile);
                }
            });
        }

        loadContextActionLibrary();
    }

    @Override
    public void onDestroy() {
        Log.e("TEST", "onDestroy");
        super.onDestroy();
        if (loader != null) {
            loader.stopDetection();
        }
    }

    private RequestResult handleRequest(RequestConfig config) {
        // TODO: handle getWindow
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
            result.putObject("getWindows", getWindows());
        }

        return result;
    }

    private void loadContextActionLibrary() {
        NetworkUtils.downloadFile(this, "classes.dex", new FileCallback() {
            @Override
            public void onSuccess(Response<File> response) {
                File file = response.body();
                File saveFile = new File(BuildConfig.SAVE_PATH, "classes.dex");
                FileUtils.copy(file, saveFile);

                final File tmpDir = getDir("dex", 0);
                classLoader = new DexClassLoader(BuildConfig.SAVE_PATH + "classes.dex", tmpDir.getAbsolutePath(), null, this.getClass().getClassLoader());
                loader = new ContextActionLoader(mContext, classLoader);

                // TapTapAction
                ActionConfig tapTapConfig = new ActionConfig();
                tapTapConfig.setAction(BuiltInActionEnum.TapTap);
                tapTapConfig.putValue("SeqLength", 50);
                tapTapConfig.setSensorType(Arrays.asList(SensorType.IMU));

                ActionListener actionListener = action ->
                        mHandler.post(() -> Toast.makeText(mContext, action.getAction(), Toast.LENGTH_SHORT).show());

                // ProximityContext
                ContextConfig proximityConfig = new ContextConfig();
                proximityConfig.setContext(BuiltInContextEnum.Proximity);
                proximityConfig.setSensorType(Arrays.asList(SensorType.PROXIMITY));

                // InformationalContext
                ContextConfig informationalConfig = new ContextConfig();
                informationalConfig.setContext(BuiltInContextEnum.Informational);
                informationalConfig.setSensorType(Arrays.asList(SensorType.ACCESSIBILITY, SensorType.BUTTON_ACTION));

                ContextListener contextListener = context ->
                        mHandler.post(() -> Toast.makeText(mContext, context.getContext(), Toast.LENGTH_SHORT).show());

                RequestListener requestListener = config -> handleRequest(config);
                loader.startDetection(Arrays.asList(tapTapConfig), actionListener, Arrays.asList(proximityConfig, informationalConfig), contextListener, requestListener);
            }
        });
    }

    class HomeWatcherReceiver extends BroadcastReceiver {
        public static final String SYSTEM_DIALOG_REASON_KEY = "reason";
        public static final String SYSTEM_DIALOG_REASON_HOME_KEY = "homekey";
        public static final String SYSTEM_DIALOG_REASON_RECENT_APPS = "recentapps";
        @Override
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (action.equals(Intent.ACTION_CLOSE_SYSTEM_DIALOGS)) {
                String reason = intent.getStringExtra(SYSTEM_DIALOG_REASON_KEY);
                if (SYSTEM_DIALOG_REASON_HOME_KEY.equals(reason)) {
                    loader.onButtonActionEvent(new ButtonActionEvent("home","global"));
                }
                if (SYSTEM_DIALOG_REASON_RECENT_APPS.equals(reason)) {
                    loader.onButtonActionEvent(new ButtonActionEvent("recentapps","global"));
                }
            }
            /*
            else if(action.equals(Intent.ACTION_SCREEN_ON))
                loader.onScreenState(true);
            else if(action.equals(Intent.ACTION_SCREEN_OFF))
                loader.onScreenState(false);
             */
        }
    }
}