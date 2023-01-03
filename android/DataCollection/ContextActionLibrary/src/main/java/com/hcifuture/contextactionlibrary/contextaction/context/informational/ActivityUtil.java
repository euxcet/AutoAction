package com.hcifuture.contextactionlibrary.contextaction.context.informational;

import android.content.ComponentName;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;

public class ActivityUtil {
    static Context context;


    public ActivityUtil(Context context)
    {
        this.context = context;
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
