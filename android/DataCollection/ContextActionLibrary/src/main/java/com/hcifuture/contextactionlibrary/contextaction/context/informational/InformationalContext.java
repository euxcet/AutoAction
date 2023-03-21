package com.hcifuture.contextactionlibrary.contextaction.context.informational;

import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.contextaction.ContextActionContainer;
import com.hcifuture.contextactionlibrary.contextaction.context.BaseContext;
import com.hcifuture.contextactionlibrary.contextaction.context.ConfigContext;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.contextactionlibrary.status.Heart;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.contextactionlibrary.contextaction.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ContextResult;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import javax.crypto.spec.PSource;

public class InformationalContext extends BaseContext {
    private static final String TAG = "TaskContext";
    private Map<String, List<Task>> tasks = new HashMap<>();
    private List<Page> pageList = new ArrayList<>();
    private List<Action> actionList = new ArrayList<Action>();

    private ActivityUtil activityUtil;
    private EventAnalyzer eventAnalyzer;

    private String lastPackageName = "";
    private String lastActivityName = "";
    private Task lastTask = null;
    private Page lastPage = null;
    private boolean windowStable = false;

    private long lastIMUTime = 0;
    private long lastActionTime = 0;

    private long lastWindowChange = 0;
    private ThreadPoolExecutor threadPoolExecutor;

    private long lastEventTime = 0;

    @RequiresApi(api = Build.VERSION_CODES.N)
    public InformationalContext(Context context, ContextConfig config, RequestListener requestListener, List<ContextListener> contextListener, LogCollector informationalLogCollector, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, config, requestListener, contextListener, scheduledExecutorService, futureList);
        logCollector = informationalLogCollector;
        activityUtil = new ActivityUtil(context);
        eventAnalyzer = new EventAnalyzer();
        threadPoolExecutor = new ThreadPoolExecutor(1, 1, 1000, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(10), Executors.defaultThreadFactory(), new ThreadPoolExecutor.DiscardOldestPolicy());

        futureList.add(scheduledExecutorService.schedule(() -> {
            eventAnalyzer.initialize(context);
            PageController.initPages(context);
            initFromFile();
        }, 0, TimeUnit.MILLISECONDS));

    }

    @Override
    public void start() {

    }

    @Override
    public void stop() {
        threadPoolExecutor.shutdown();
    }

    @Override
    public void onIMUSensorEvent(SingleIMUData data) {
        lastIMUTime = data.getTimestamp();
    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {

    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public void onAction(Action action) {
        long nowTime = System.currentTimeMillis();
        addTaskLog(new LogItem(action.toString(),"action",new Date()));
        if(action.getType().equals("text_change"))
            action.setText("EDIT_TEXT");

        actionList.add(action);

        if (nowTime - lastActionTime > 5000) {
            if (contextListener != null) {
                for (ContextListener listener: contextListener) {
                    // need imu record
                    ContextResult contextResult = new ContextResult("UserAction");
                    contextResult.setTimestamp(lastIMUTime);
                    listener.onContext(contextResult);
                    // need scan
                    ContextResult contextResultScan = new ContextResult(ConfigContext.NEED_SCAN, "UserAction");
                    contextResultScan.setTimestamp(nowTime);
                    listener.onContext(contextResultScan);
                }
            }
        }

        lastActionTime = nowTime;
    }

    public void onScreenState(boolean screen_on)
    {
        addTaskLog(new LogItem(String.valueOf(screen_on),"screen",new Date()));
    }

    public void onActivityChange(String activity)
    {
        addTaskLog(new LogItem(activity,"activity",new Date()));
    }

    public void onPackageChange(String packageName)
    {
        addTaskLog(new LogItem(packageName,"package",new Date()));
    }


    @RequiresApi(api = Build.VERSION_CODES.N)
    public void onWindowStable()
    {
        final long timestamp = System.currentTimeMillis();
        if (timestamp - lastWindowChange < 100)
            return;
        this.lastWindowChange = timestamp;
        final String lastActivityName = this.lastActivityName;
        final String lastPackageName = this.lastPackageName;
        final List<AccessibilityNodeInfo> nodeInfos = new ArrayList<>();
        for(AccessibilityNodeInfo root:AccessibilityNodeInfoRecordFromFile.getAllRoots(requestListener))
            nodeInfos.add(AccessibilityNodeInfo.obtain(root));
        final Date date = new Date();
//        futureList.add(scheduledExecutorService.schedule(() -> {
        // change to one thread
        threadPoolExecutor.execute(() -> {
                    HashSet<String> allFunctionWords = PageController.getAllFunctionWords(AccessibilityNodeInfoRecordFromFile.buildAllTrees(nodeInfos, lastActivityName));
                    addTaskLog(new LogItem(allFunctionWords.toString(), "functionWords", date));
                    Page page = PageController.recognizePage(allFunctionWords, lastPackageName);
                    if (page != null) {
                        // 从页面端不重复记录
                        if (lastPage != null && lastPage.getId() == page.getId())
                            return;
//                Log.d("InformationalContext","page match" + page.getTitle());
                        addTaskLog(new LogItem(page.getTitle(), "page", date));
                        pageList.add(page);
                        Task task = recognizeTask();
                        if (task != null) {
//                    Log.d("InformationalContext","task match" + task.getName());
                            addTaskLog(new LogItem(task.getName(), "task", date));
                        }
                        lastTask = task;
                    }
                    lastPage = page;
                });
//        },0,TimeUnit.MILLISECONDS));
    }

    private Task recognizeTask() {
        int p_size = pageList.size();
        if(p_size>20)
            pageList = pageList.subList(p_size-10,p_size);
        int a_size = actionList.size();
        if(a_size>20)
            actionList = actionList.subList(a_size-10,a_size);

        if(tasks.containsKey(lastPackageName)) {
            for (Task task : tasks.get(lastPackageName)) {
                if (task.match(pageList,actionList))
                    return task;
            }
        }
        return null;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        // remove duplicated
        if(event.getEventTime()==lastEventTime)
            return;

        Heart.getInstance().newContextAliveEvent(getConfig().getContext(), event.getEventTime());

        lastEventTime = event.getEventTime();

        String eventString =event.toString();
        long eventTime  = System.currentTimeMillis();
        final String eventStr = ("timeStamp:"+eventTime+";"+eventString).replace("\n"," ");

        if(event.getEventType()==AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED) {
            eventAnalyzer.add(eventStr);
            return;
        }

        int eventType = event.getEventType();

        if(eventType==AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) {
            String packageName = event.getPackageName().toString();
            String activityName = ActivityUtil.getActivityName(packageName, event.getClassName(), lastActivityName);
            if(activityName!=null&&(lastActivityName==null || !lastActivityName.equals(activityName)))
            {
                onActivityChange(activityName);
                lastActivityName = activityName;
            }
            if(activityName!=null)
                packageName = activityName.split("/")[0];
            if(packageName!=null&&(lastPackageName==null || !lastPackageName.equals(packageName)))
            {
                onPackageChange(packageName);
                lastPackageName = packageName;
            }
        }

        if(eventType == AccessibilityEvent.TYPE_VIEW_CLICKED ||
                eventType == AccessibilityEvent.TYPE_VIEW_LONG_CLICKED ||
                eventType == AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED)
        {
            if(!windowStable) {
                onWindowStable();
            }
            String text = "";
            try {
                AccessibilityNodeInfo source = null;
                if (event.getSource() != null)
                    source = AccessibilityNodeInfo.obtain(event.getSource());
                if(source!=null) {
                    if (source.getClassName() != null)
                        text = "class:" + source.getClassName().toString();
                    if (source.getText() != null)
                        text = "text:" + source.getText().toString();
                    if (source.getContentDescription() != null)
                        text += ",cd:" + source.getContentDescription().toString();
                }
            }catch (Exception e)
            {
                Log.e("InformationalContext",eventString+"error on source");
                e.printStackTrace();
            }
            if(event.getText()!=null&&!event.getText().isEmpty() && text.equals(""))
                text += ",event-text:"+event.getText().get(0).toString();
            if(event.getContentDescription()!=null)
                text += ",event-cd:"+event.getContentDescription().toString();
            Action ac = new Action(text,eventTypeToString(event.getEventType()));
            onAction(ac);
            windowStable = false;
        }

        float model_result = eventAnalyzer.analyze(eventStr);
        addTaskLog(new LogItem(eventString+"--"+model_result,"accessibilityEvent",new Date()));
        Log.i("InformationalContext",eventString+"\n"+model_result);

        if(model_result>0.5 && !windowStable)
        {
            onWindowStable();
            windowStable = true;
        }
        if(model_result<=0.5)
            windowStable = false;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onBroadcastEvent(BroadcastEvent event) {
        String action = event.getAction();
        if (action.equals(Intent.ACTION_CLOSE_SYSTEM_DIALOGS)) {
//            String reason = event.getTag();
            String reason = event.getExtras().getString("reason");
            if (SYSTEM_DIALOG_REASON_HOME_KEY.equals(reason)) {
                onAction(new Action("home","global"));
            }
            if (SYSTEM_DIALOG_REASON_RECENT_APPS.equals(reason)) {
                onAction(new Action("recentapps","global"));
            }
        }
        else if(action.equals(Intent.ACTION_SCREEN_ON))
            onScreenState(true);
        else if(action.equals(Intent.ACTION_SCREEN_OFF))
            onScreenState(false);
    }

    @Override
    public void onExternalEvent(Bundle bundle) {

    }

    @Override
    public void getContext() {

    }

    @Override
    public String getName() {
        return "InformationalContext";
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public void initFromFile()
    {
        try
        {
            InputStream inStream = new FileInputStream(ContextActionContainer.getSavePath() + "tasks.csv");
            BufferedReader br = new BufferedReader(new InputStreamReader(inStream));
            String line;
            while ((line = br.readLine()) != null) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                    loadLine(line);
                }
            }
            br.close();
            inStream.close();
        }catch(Exception e)
        {
            e.printStackTrace();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public void loadLine(String line)
    {
        if (line.startsWith("id"))
            return;
        String obj[] = line.split(";");
        if (obj.length != 5&&obj.length!=4)
            return;
        try {
            String id = obj[0];
            String text = obj[1];
            String pkg = obj[2];
            String pages = obj[3];
            String actions;
            if(obj.length==4)
                actions = "";
            else
                actions = obj[4];

            String[] ps = pages.split(",");
            List<Page> pl = new ArrayList<>();
            for(String p:ps) {
                Page tp=PageController.getIdToPage().get(Integer.parseInt(p));
                if(tp==null) {
                    Log.e("InformationalContext","page not exist"+p);
                    return;
                }
                pl.add(tp);
            }

            String[] as = actions.split("##");
            List<Action> al = new ArrayList<>();
            if(obj.length!=4) {
                for (String a : as) {
                    a = a.substring(1, a.length() - 1);
                    JSONObject aj = null;
                    aj = new JSONObject(a);
                    al.add(new Action(aj.getString("param"), aj.getString("typeString")));
                }
            }
            Task task = new Task(Integer.parseInt(id),text,pkg,pl,al);
            List<Task> list = tasks.getOrDefault(pkg, new ArrayList<>());
            list.add(task);
            tasks.put(pkg, list);
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    public static final String SYSTEM_DIALOG_REASON_KEY = "reason";
    public static final String SYSTEM_DIALOG_REASON_HOME_KEY = "homekey";
    public static final String SYSTEM_DIALOG_REASON_RECENT_APPS = "recentapps";

    public static String eventTypeToString(int eventType)
    {
        switch (eventType)
        {
            case  AccessibilityEvent.TYPE_VIEW_CLICKED:
                return "click";
            case AccessibilityEvent.TYPE_VIEW_LONG_CLICKED:
                return "long_click";
            case AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED:
                return "text_change";
        }
        return "";
    }

    public void addTaskLog(LogItem item) {
        StringBuilder sb = new StringBuilder("");
        sb.append(item.task);
        sb.append("#");
        sb.append(item.type);
        sb.append("#");
        sb.append(LogItem.formatter.format(item.getTime()));
        Log.d("InformationalContext",sb.toString());
        if (logCollector != null) {
            logCollector.addLog(sb.toString());
        }
    }

}
