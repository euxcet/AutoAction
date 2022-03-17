package com.example.datacollection.ui;

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
import com.example.ncnnlibrary.communicate.event.BroadcastEvent;
import com.example.ncnnlibrary.communicate.listener.ActionListener;
import com.example.ncnnlibrary.communicate.listener.ContextListener;
import com.example.ncnnlibrary.communicate.listener.RequestListener;
import com.example.ncnnlibrary.communicate.result.RequestResult;
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
        Log.e("TEST", "onServiceConnected");

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