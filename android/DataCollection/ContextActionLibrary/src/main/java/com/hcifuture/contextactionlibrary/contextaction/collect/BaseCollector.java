package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;

import com.hcifuture.contextactionlibrary.collect.trigger.ClickTrigger;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.Enumeration;

public abstract class BaseCollector {
    protected Context mContext;
    protected RequestListener requestListener;
    protected ClickTrigger clickTrigger;

    public BaseCollector(Context context, RequestListener requestListener, ClickTrigger clickTrigger) {
        this.mContext = context;
        this.requestListener = requestListener;
        this.clickTrigger = clickTrigger;
    }

    public abstract void onAction(ActionResult action);

    public abstract void onContext(ContextResult context);

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
