package com.example.simpleexample;

import android.accessibilityservice.AccessibilityService;
import android.content.Context;
import android.content.Intent;
import android.os.Handler;
import android.os.Looper;
import android.view.accessibility.AccessibilityEvent;
import android.widget.Toast;

import com.example.simpleexample.contextaction.LoaderManager;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

public class MainService extends AccessibilityService implements ContextListener, ActionListener {
    private Context mContext;
    private Handler mHandler;

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
        this.mContext = this;
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
            /*
            if (action.getAction().equals("TapTap")) {
                loaderManager.upgrade();
            }
             */
        });
    }

    @Override
    public void onContext(ContextResult context) {
        mHandler.post(() -> Toast.makeText(mContext, context.getContext(), Toast.LENGTH_SHORT).show());
    }
}