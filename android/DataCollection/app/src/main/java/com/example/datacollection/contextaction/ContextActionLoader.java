package com.example.datacollection.contextaction;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import dalvik.system.DexClassLoader;

public class ContextActionLoader {
    private Context mContext;
    private ClassLoader classLoader;

    private Class actionConfigClass;
    private Class actionListenerClass;
    private Class containerClass;

    public ContextActionLoader(Context context, DexClassLoader classLoader) {
        this.mContext = context;
        this.classLoader = classLoader;
        try {
            containerClass = classLoader.loadClass("com.example.contextactionlibrary.contextaction.ContextActionContainer");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Object newContainer(List<String> actions) {
        try {
            return containerClass.getDeclaredConstructor(Context.class, List.class, boolean.class).newInstance(mContext, actions, true);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private void startContainer(Object container) {
        try {
            Method start = containerClass.getMethod("start");
            start.invoke(container);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void load() {
        try {
            Object container = newContainer(Arrays.asList("TapTap"));
            startContainer(container);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

