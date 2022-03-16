package com.example.contextactionlibrary.contextaction.context.informational;

import android.content.ComponentName;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;

import java.util.HashMap;
import java.util.HashSet;

public class ActivityUtil {
    static Context context;
    public static HashMap<String,String> packageNameToApp= new HashMap<>();
    public static HashSet<String> packageNameIgnore = new HashSet<>();


    public ActivityUtil(Context context)
    {
        this.context = context;
        packageNameToApp.put("com.tencent.mobileqq","qq");
        packageNameToApp.put("com.android.settings","settings");


        packageNameIgnore.add("com.lijiahui.lifelog");
        packageNameIgnore.add("com.huawei.android.launcher");
    }


    public static String getActivityName(CharSequence packagename,CharSequence classname,String lastclassname)
    {
        if(packagename==null||classname==null)
            return lastclassname;
        ComponentName componentName = new ComponentName(
                packagename.toString(),
                classname.toString()
        );
        ActivityInfo activityInfo = tryGetActivity(componentName);
        boolean isActivity = activityInfo != null;
        if (isActivity)
            return componentName.flattenToShortString();
        else
            return lastclassname;
    }

    private static ActivityInfo tryGetActivity(ComponentName componentName) {
        try {
            return context.getPackageManager().getActivityInfo(componentName, 0);
        } catch (PackageManager.NameNotFoundException e) {
            return null;
        }
    }
}
