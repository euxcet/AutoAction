package com.hcifuture.contextactionlibrary.contextaction.context.informational;

import android.content.Context;
import android.content.Intent;
import android.hardware.SensorEvent;
import android.os.Build;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.contextaction.context.BaseContext;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.shared.communicate.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class InformationalContext extends BaseContext {
    private static final String TAG = "TaskContext";
    private Map<String, List<Task>> tasks = new HashMap<>();
    private List<Page> pageList = new ArrayList<>();
    private List<BroadcastEvent> actionList = new ArrayList<>();

    private ActivityUtil activityUtil;
    private EventAnalyzer eventAnalyzer;
    private PageController pageController;

    private String lastPackageName = "";
    private String lastActivityName = "";
    private Task lastTask = null;

    private boolean windowStable = false;

    public InformationalContext(Context context, ContextConfig config, RequestListener requestListener, List<ContextListener> contextListener) {
        super(context, config, requestListener, contextListener);
        activityUtil = new ActivityUtil(context);
        eventAnalyzer = new EventAnalyzer();
        eventAnalyzer.initialize(context);

        pageController = new PageController(context);
        List<Task> tencentTask = new ArrayList<>();
        List<Page> page = new ArrayList<>();
        page.add(pageController.getPages().get("com.tencent.mm").get(0));
        tencentTask.add(new Task(0,"查看朋友圈","查看朋友圈",page,new ArrayList<>()));
        tasks.put("com.tencent.mm",tencentTask);
    }

    @Override
    public void start() {

    }

    @Override
    public void stop() {

    }

    @Override
    public void onIMUSensorChanged(SensorEvent event) {

    }

    @Override
    public void onProximitySensorChanged(SensorEvent event) {

    }

    private void onActivityChange(String activityName) {
    }

    private void onPackageChange(String packageName) {
        lastPackageName = packageName;
    }

    private void onScreenState(boolean screenOn) {

    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public void onWindowStable() {
        Page page = pageController.recognizePage(AccessibilityNodeInfoRecordFromFile.buildAllTrees(requestListener, lastPackageName), lastPackageName);
        if (page != null) {
            Log.e(TAG, "Page match " + page.getTitle());
            pageList.add(page);
            Task task = recognizeTask();
            if (task != lastTask) {
                lastTask = task;
                Log.e(TAG, "Task match " + task.getDescribe());
            }
        }
    }

    private Task recognizeTask() {
        int p_size = pageList.size();
        if (p_size > 20) {
            pageList = pageList.subList(p_size - 10, p_size);
        }
        int a_size = actionList.size();
        if (a_size > 20) {
            actionList = actionList.subList(a_size - 10, a_size);
        }

        if (tasks.containsKey(lastPackageName)) {
            for (Task task : tasks.get(lastPackageName)) {
                if (task.match(pageList, actionList))
                    return task;
            }
        }
        return null;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        String eventString = event.toString();
        long eventTime = System.currentTimeMillis();
        final String eventStr = ("timeStamp:" + eventTime + ";" + eventString).replace("\n", " ");

        int eventType = event.getEventType();

        if (eventType == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) {
            String packageName = event.getPackageName().toString();
            String activityName = activityUtil.getActivityName(packageName, event.getClassName(), lastActivityName);
            if (lastPackageName != null && (lastActivityName == null || !lastActivityName.equals(activityName))) {
                onActivityChange(activityName);
                lastActivityName = activityName;
            }

            if (packageName != null && (lastPackageName == null || !lastPackageName.equals(packageName))) {
                onPackageChange(packageName);
                lastPackageName = packageName;
            }

        }

        AccessibilityEvent eventRecord = AccessibilityEvent.obtain(event);

        if (eventType == AccessibilityEvent.TYPE_VIEW_CLICKED ||
                eventType == AccessibilityEvent.TYPE_VIEW_LONG_CLICKED ||
                eventType == AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED) {
            String text = "";
            if (event.getText() != null && !event.getText().isEmpty())
                text = event.getText().get(0).toString();
            if (event.getContentDescription() != null && text.equals(""))
                text = event.getContentDescription().toString();
            actionList.add(new BroadcastEvent("", text, String.valueOf(event.getEventType())));
            windowStable = false;
        }

        float model_result = eventAnalyzer.analyze(eventStr);
        Log.e("model_result:", eventString + "\n" + model_result);

        if (model_result > 0.5 && !windowStable) {
            onWindowStable();
            windowStable = true;
        }

        eventRecord.recycle();
    }

    @Override
    public void onBroadcastEvent(BroadcastEvent event) {
        if (event.getAction().equals(Intent.ACTION_CLOSE_SYSTEM_DIALOGS)) {
            actionList.add(event);
        }
    }

    @Override
    public void getContext() {

    }
}
