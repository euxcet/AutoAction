package com.hcifuture.datacollection.service;

import android.accessibilityservice.AccessibilityService;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.ContentObserver;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.view.KeyEvent;
import android.view.accessibility.AccessibilityEvent;
import android.widget.Toast;

import androidx.annotation.Nullable;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.NcnnInstance;
import com.hcifuture.datacollection.contextaction.ContextActionLoader;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.hcifuture.shared.NcnnFunction;
import com.hcifuture.shared.communicate.BuiltInActionEnum;
import com.hcifuture.shared.communicate.BuiltInContextEnum;
import com.hcifuture.shared.communicate.SensorType;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.shared.communicate.config.RequestConfig;
import com.hcifuture.shared.communicate.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.RequestResult;
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

    private CustomBroadcastReceiver mBroadcastReceiver;

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
        mBroadcastReceiver = new CustomBroadcastReceiver();
        final IntentFilter filter = new IntentFilter();
        filter.addAction(Intent.ACTION_CLOSE_SYSTEM_DIALOGS);
        filter.addAction(Intent.ACTION_SCREEN_ON);
        filter.addAction(Intent.ACTION_SCREEN_OFF);
        filter.addAction(Intent.ACTION_AIRPLANE_MODE_CHANGED);
        filter.addAction(Intent.ACTION_CONFIGURATION_CHANGED);
        filter.addAction(Intent.ACTION_MEDIA_BUTTON);
        filter.addAction(Intent.ACTION_SET_WALLPAPER);
        registerReceiver(mBroadcastReceiver, filter);

        FileUtils.downloadFiles(this, Arrays.asList(
                "param_dicts.json",
                "param_max.json",
                "words.csv",
                "best.bin",
                "best.param",
                "classes.dex"
        ), this::loadContextActionLibrary);
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

        // TableContext
        ContextConfig tableConfig = new ContextConfig();
        tableConfig.setContext(BuiltInContextEnum.Table);
        tableConfig.setSensorType(Arrays.asList(SensorType.IMU));

        // InformationalContext
        ContextConfig informationalConfig = new ContextConfig();
        informationalConfig.setContext(BuiltInContextEnum.Informational);
        informationalConfig.setSensorType(Arrays.asList(SensorType.ACCESSIBILITY, SensorType.BROADCAST));

        ContextListener contextListener = context ->
                mHandler.post(() -> Toast.makeText(mContext, context.getContext(), Toast.LENGTH_SHORT).show());

        RequestListener requestListener = config -> handleRequest(config);
        loader.startDetection(Arrays.asList(tapTapConfig), actionListener, Arrays.asList(proximityConfig, tableConfig, informationalConfig), contextListener, requestListener);


        NcnnInstance.init(mContext,
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
    }

    class CustomBroadcastReceiver extends BroadcastReceiver {
        public static final String ACTION_TYPE_GLOBAL = "global";

        @Override
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            switch (action) {
                case Intent.ACTION_CLOSE_SYSTEM_DIALOGS:
                    loader.onBroadcastEvent(new BroadcastEvent(
                            action,
                            intent.getStringExtra("reason"),
                            ACTION_TYPE_GLOBAL));
                    break;
                default:
                    loader.onBroadcastEvent(new BroadcastEvent(
                            action,
                            "",
                            ACTION_TYPE_GLOBAL));
                    break;
            }
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
            String key;
            String tag;
            int value = 0;

            if (uri == null) {
                key = "null";
                tag = "Unknown content change";
            } else {
                key = uri.toString();
                String database_key = uri.getLastPathSegment();
                String inter = uri.getPathSegments().get(0);
                if (inter.equals("system")) {
                    value = Settings.System.getInt(getContentResolver(), database_key, value);
                } else if (inter.equals("global")) {
                    value = Settings.Global.getInt(getContentResolver(), database_key, value);
                }
                tag = database_key;
            }

            loader.onBroadcastEvent(new BroadcastEvent(key, tag, "", value));
        }
    }

    @Override
    protected boolean onKeyEvent(KeyEvent event) {
        loader.onBroadcastEvent(new BroadcastEvent("KeyEvent", "", "", event.getKeyCode()));
        return super.onKeyEvent(event);
    }
}