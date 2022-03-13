package com.example.datacollection.ui;

import android.accessibilityservice.AccessibilityService;
import android.app.Notification;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;
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

    public MainService() {
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent accessibilityEvent) {
        Log.e("Test", "onAccessibilityEvent");
    }

    @Override
    public void onInterrupt() {

    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.e("Test", "onStartCommand");
        return super.onStartCommand(intent, flags, startId);
    }

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        Log.e("Test", "onServiceConnected");
        mContext = this;
        mHandler = new Handler(Looper.getMainLooper());
        loadContextActionLibrary();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (loader != null) {
            loader.stopDetection();
        }
    }

    private RequestResult handleRequest(RequestConfig config) {
        RequestResult result = new RequestResult();
        String packageName = config.getString("getAppTapBlockValue");
        if (packageName != null) {
            result.putValue("getAppTapBlockValueResult", 0);
        }

        Boolean request = config.getBoolean("getCanTapTap");
        if (request != null) {
            result.putValue("getCanTapTapResult", true);
        }

        request = config.getBoolean("getCanTopTap");
        if (request != null) {
            result.putValue("getCanTopTapResult", true);
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

                ActionConfig tapTapConfig = new ActionConfig();
                tapTapConfig.setAction(BuiltInActionEnum.TapTap);
                tapTapConfig.putValue("SeqLength", 50);
                tapTapConfig.setSensorType(Arrays.asList(SensorType.IMU));


                ActionListener actionListener = action ->
                        mHandler.post(() -> Toast.makeText(mContext, action.getAction(), Toast.LENGTH_SHORT).show());

                ContextConfig proximityConfig = new ContextConfig();
                proximityConfig.setContext(BuiltInContextEnum.Proximity);
                proximityConfig.setSensorType(Arrays.asList(SensorType.PROXIMITY));

                ContextListener contextListener = context ->
                        mHandler.post(() -> Toast.makeText(mContext, context.getContext(), Toast.LENGTH_SHORT).show());

                RequestListener requestListener = config -> handleRequest(config);
                loader.startDetection(Arrays.asList(tapTapConfig), actionListener, Arrays.asList(proximityConfig), contextListener, requestListener);
            }
        });
    }
}