package com.hcifuture.datacollection.service;

import android.accessibilityservice.AccessibilityService;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.KeyEvent;
import android.view.accessibility.AccessibilityEvent;
import android.widget.Toast;

import com.amap.api.services.core.ServiceSettings;
import com.hcifuture.datacollection.contextaction.LoaderManager;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;

public class MainService extends AccessibilityService implements ContextListener, ActionListener {
    private Context mContext;
    private Handler mHandler;

    private static MainService self;

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
        self = this;
        mContext = this;
        this.mHandler = new Handler(Looper.getMainLooper());
        this.loaderManager = new LoaderManager(this, this, this);
        loaderManager.start();

        Log.e("Location", sHA1(getApplicationContext()));
    }

    @Override
    public boolean onUnbind(Intent intent) {
        if (loaderManager != null) {
            loaderManager.stop();
        }

        self = null;

        return super.onUnbind(intent);
    }

    @Override
    public void onAction(ActionResult action) {
        mHandler.post(() -> {
            Toast.makeText(mContext, action.getAction(), Toast.LENGTH_SHORT).show();
        });
    }

    @Override
    public void onContext(ContextResult context) {
        mHandler.post(() -> Toast.makeText(mContext, context.getContext(), Toast.LENGTH_SHORT).show());
    }

    @Override
    protected boolean onKeyEvent(KeyEvent event) {
        if (loaderManager != null) {
            loaderManager.onKeyEvent(event);
        }
        return super.onKeyEvent(event);
    }

    public static String sHA1(Context context){
        try {
            PackageInfo info = context.getPackageManager().getPackageInfo(
                    context.getPackageName(), PackageManager.GET_SIGNATURES);
            byte[] cert = info.signatures[0].toByteArray();
            MessageDigest md = MessageDigest.getInstance("SHA1");
            byte[] publicKey = md.digest(cert);
            StringBuffer hexString = new StringBuffer();
            for (int i = 0; i < publicKey.length; i++) {
                String appendString = Integer.toHexString(0xFF & publicKey[i])
                        .toUpperCase(Locale.US);
                if (appendString.length() == 1)
                    hexString.append("0");
                hexString.append(appendString);
                hexString.append(":");
            }
            String result = hexString.toString();
            return result.substring(0, result.length()-1);
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static MainService getInstance() {
        return self;
    }

    public void upgrade() {
        loaderManager.upgrade();
    }
}