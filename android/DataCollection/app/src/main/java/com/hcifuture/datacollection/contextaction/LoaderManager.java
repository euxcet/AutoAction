package com.hcifuture.datacollection.contextaction;

import static android.content.Context.MODE_PRIVATE;

import android.accessibilityservice.AccessibilityService;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.accessibility.AccessibilityEvent;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.shared.communicate.config.RequestConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.RequestResult;
import com.hcifuture.shared.communicate.status.Heartbeat;

import java.io.File;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
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

        if (config.getString("getDeviceId") != null) {
            String mac = getMacMoreThanM();
            if (mac == null) {
                result.putObject("getDeviceId", "NULL");
            } else {
                result.putObject("getDeviceId", mac.replace(":", "_"));
            }
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

        RequestListener requestListener = this::handleRequest;

        loader.startDetection(actionListener, contextListener, requestListener);

        final boolean[] enable = {true};
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                Bundle bundle = new Bundle();
                ArrayList<String> list = new ArrayList<>();
//                list.add("TapTapAction");
                list.add("TopTapAction");
                if (enable[0]) {
                    Log.e("TEST", "Disable!");
                    bundle.putStringArrayList("DisableFunctions", list);
                    loader.onExternalEvent(bundle);
                    enable[0] = false;
                } else {
                    Log.e("TEST", "Enable!");
                    bundle.putStringArrayList("EnableFunctions", list);
                    loader.onExternalEvent(bundle);
                    enable[0] = true;
                }
            }
        }, 8000, 8000);
    }

    public void onAccessibilityEvent(AccessibilityEvent event) {
        if (loader != null) {
            loader.onAccessibilityEvent(event);
        }
    }

    public void onKeyEvent(KeyEvent event) {
        if (loader != null) {
            loader.onKeyEvent(event);
        }
    }

    protected static String getMacMoreThanM() {
        try {
            Enumeration enumeration = NetworkInterface.getNetworkInterfaces();
            while (enumeration.hasMoreElements()) {
                NetworkInterface networkInterface = (NetworkInterface) enumeration.nextElement();

                byte[] arrayOfByte = networkInterface.getHardwareAddress();
                if (arrayOfByte == null || arrayOfByte.length == 0) {
                    continue;
                }

                StringBuilder stringBuilder = new StringBuilder();
                for (byte b : arrayOfByte) {
                    stringBuilder.append(String.format("%02X:", b));
                }
                if (stringBuilder.length() > 0) {
                    stringBuilder.deleteCharAt(stringBuilder.length() - 1);
                }
                String str = stringBuilder.toString();
                if (networkInterface.getName().equals("wlan0")) {
                    return str;
                }
            }
        } catch (SocketException socketException) {
            return null;
        }
        return null;
    }
}
