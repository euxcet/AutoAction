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
import com.hcifuture.datacollection.contextaction.LoaderManager;
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
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;
import com.hcifuture.shared.communicate.result.RequestResult;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.Arrays;
import java.util.Timer;
import java.util.TimerTask;

import dalvik.system.DexClassLoader;

public class MainService extends AccessibilityService implements ContextListener, ActionListener {
    private Context mContext;
    private Handler mHandler;

    private CustomBroadcastReceiver mBroadcastReceiver;

    private LoaderManager loaderManager;

    public MainService() {
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        if (loaderManager != null) {
            loaderManager.onAccessibilityEvent(event);
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

        this.mHandler = new Handler(Looper.getMainLooper());
        this.loaderManager = new LoaderManager(this, this, this);
        loaderManager.start();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (loaderManager != null) {
            loaderManager.stop();
        }
    }

    @Override
    public void onAction(ActionResult action) {
        mHandler.post(() -> {
            Toast.makeText(mContext, action.getAction(), Toast.LENGTH_SHORT).show();
            // loaderManager.upgrade();
        });
    }

    @Override
    public void onContext(ContextResult context) {
        mHandler.post(() -> Toast.makeText(mContext, context.getContext(), Toast.LENGTH_SHORT).show());
    }

    class CustomBroadcastReceiver extends BroadcastReceiver {
        public static final String ACTION_TYPE_GLOBAL = "global";

        @Override
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            switch (action) {
                case Intent.ACTION_CLOSE_SYSTEM_DIALOGS:
                    if (loaderManager != null) {
                        loaderManager.onBroadcastEvent(new BroadcastEvent(
                                action,
                                intent.getStringExtra("reason"),
                                ACTION_TYPE_GLOBAL));
                    }
                    break;
                default:
                    if (loaderManager != null) {
                        loaderManager.onBroadcastEvent(new BroadcastEvent(
                                action,
                                "",
                                ACTION_TYPE_GLOBAL));
                    }
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

            if (loaderManager != null) {
                loaderManager.onBroadcastEvent(new BroadcastEvent(key, tag, "", value));
            }
        }
    }

    @Override
    protected boolean onKeyEvent(KeyEvent event) {
        if (loaderManager != null) {
            loaderManager.onBroadcastEvent(new BroadcastEvent("KeyEvent", "", "", event.getKeyCode()));
        }
        return super.onKeyEvent(event);
    }
}