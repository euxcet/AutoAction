package com.example.simpleexample;

import android.accessibilityservice.AccessibilityService;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.ContentObserver;
import android.net.Uri;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.view.KeyEvent;
import android.view.accessibility.AccessibilityEvent;
import android.widget.Toast;

import androidx.annotation.Nullable;

import com.example.simpleexample.contextaction.ContextActionLoader;
import com.example.simpleexample.utils.FileUtils;
import com.example.simpleexample.utils.NetworkUtils;
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
import java.security.AlgorithmParameterGenerator;
import java.util.Arrays;

import dalvik.system.DexClassLoader;

public class MainService extends AccessibilityService {
    private Context mContext;
    private DexClassLoader classLoader;
    private ContextActionLoader loader;
    private Handler mHandler;

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
}