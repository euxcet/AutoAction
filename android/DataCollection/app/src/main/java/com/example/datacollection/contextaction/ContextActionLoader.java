package com.example.datacollection.contextaction;

import android.content.Context;
import android.widget.Toast;

import com.example.ncnnlibrary.communicate.ActionConfig;
import com.example.ncnnlibrary.communicate.ActionListener;
import com.example.ncnnlibrary.communicate.ActionResult;
import com.example.ncnnlibrary.communicate.BuiltInActionEnum;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import dalvik.system.DexClassLoader;

public class ContextActionLoader {
    private Context mContext;
    private ClassLoader classLoader;

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

    private Object newContainer(List<ActionConfig> config, ActionListener listener) {
        try {
            return containerClass.getDeclaredConstructor(Context.class, List.class, ActionListener.class, boolean.class).newInstance(mContext, config, listener, true);
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

    public void startDetection(List<ActionConfig> config, ActionListener actionListener) {
        try {
            Object container = newContainer(config, actionListener);
            startContainer(container);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

