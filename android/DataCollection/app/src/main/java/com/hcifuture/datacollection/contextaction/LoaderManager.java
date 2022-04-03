package com.hcifuture.datacollection.contextaction;

import static android.content.Context.MODE_PRIVATE;

import android.accessibilityservice.AccessibilityService;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;
import android.widget.Toast;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.NcnnInstance;
import com.hcifuture.datacollection.utils.FileUtils;
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
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.RequestResult;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.atomic.AtomicBoolean;

import dalvik.system.DexClassLoader;

public class LoaderManager {
    private AccessibilityService mService;
    private DexClassLoader classLoader;
    private ContextActionLoader loader;
    private final List<String> UPDATABLE_FILES = Arrays.asList(
            "param_dicts.json",
            "param_max.json",
            "words.csv",
            "best.bin",
            "best.param",
            "classes.dex",
            "tap7cls_pixel4.tflite",
            "ResultModel.tflite"
    );
    private AtomicBoolean isUpgrading;
    private ContextListener contextListener;
    private ActionListener actionListener;

    public LoaderManager(AccessibilityService service, ContextListener contextListener, ActionListener actionListener) {
        this.mService = service;
        this.isUpgrading = new AtomicBoolean(false);
        if (contextListener == null) {
            this.contextListener = (p) -> {};
        } else {
            this.contextListener = contextListener;
        }
        if (actionListener == null) {
            this.actionListener = new ActionListener() {
                @Override
                public void onActionRecognized(ActionResult action) {

                }

                @Override
                public void onAction(ActionResult action) {

                }

                @Override
                public void onActionSave(ActionResult action) {

                }
            };
        } else {
            this.actionListener = actionListener;
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
            result.putObject("getWindows", mService.getWindows());
        }

        return result;
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
        Log.e("TEST", "start");
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
            }, 2000);
        });
    }

    public void stop() {
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

        // TapTapAction
        ActionConfig tapTapConfig = new ActionConfig();
        tapTapConfig.setAction(BuiltInActionEnum.TapTap);
        tapTapConfig.putValue("SeqLength", 50);
        tapTapConfig.setSensorType(Arrays.asList(SensorType.IMU));

        // TopTapAction
        ActionConfig topTapConfig = new ActionConfig();
        topTapConfig.setAction(BuiltInActionEnum.TopTap);
        topTapConfig.putValue("SeqLength", 25);
        topTapConfig.setSensorType(Arrays.asList(SensorType.IMU));

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

        RequestListener requestListener = config -> handleRequest(config);
        loader.startDetection(Arrays.asList(tapTapConfig, topTapConfig), actionListener, Arrays.asList(proximityConfig, tableConfig, informationalConfig), contextListener, requestListener);


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
}
