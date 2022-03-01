package com.example.datacollection.utils;

import android.content.Context;
import android.util.Log;

import com.example.datacollection.TaskList;
import com.google.gson.Gson;
import com.lzy.okgo.OkGo;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class NetworkUtils {
    private static final String TAG = "NetworkUtils";
    private static final String ROOT_URL = "http://192.168.31.186:60010";
    private static final String ALL_TASKLIST_URL = ROOT_URL + "/all_taskList";
    private static final String TASKLIST_HISTORY_URL = ROOT_URL + "/taskList_history";
    private static final String TASKLIST_URL = ROOT_URL + "/taskList";
    private static final String RECORD_LIST_URL = ROOT_URL + "/record_list";
    private static final String RECORD_URL = ROOT_URL + "/record";
    private static final String RECORD_FILE_URL = ROOT_URL + "/record_file";
    private static final String SAMPLE_NUMBER_URL = ROOT_URL + "/sample_number";
    private static final String SAMPLE_URL = ROOT_URL + "/sample";
    private static final String CUTTER_TYPE_URL = ROOT_URL + "/cutter_type";
    private static final String TRAIN_URL = ROOT_URL + "/train";

    private static Gson gson = new Gson();
    /*
    private static final ThreadPoolExecutor executor = new ThreadPoolExecutor(
            1,
            2,
            10,
            TimeUnit.MINUTES,
            new LinkedBlockingQueue<Runnable>(),
            Executors.defaultThreadFactory(),
            new ThreadPoolExecutor.AbortPolicy()
            );
     */

    public static void getAllTaskList(Context context, StringCallback callback) {
        OkGo.<String>get(ALL_TASKLIST_URL)
                .tag(context)
                .execute(callback);
    }

    public static void getTaskListHistory(Context context, String taskListId, StringCallback callback) {
        OkGo.<String>get(TASKLIST_HISTORY_URL)
                .tag(context)
                .params("taskListId", taskListId)
                .execute(callback);
    }

    public static void getTaskList(Context context, String taskListId, long timestamp, StringCallback callback) {
        OkGo.<String>get(TASKLIST_URL)
                .tag(context)
                .params("taskListId", taskListId)
                .params("timestamp", timestamp)
                .execute(callback);
    }

    public static void updateTaskList(Context context, TaskList tasklist, long timestamp, StringCallback callback) {
        OkGo.<String>post(TASKLIST_URL)
                .tag(context)
                .params("taskList", gson.toJson(tasklist))
                .params("timestamp", timestamp)
                .isMultipart(true)
                .execute(callback);
    }

    public static void getRecordList(Context context, String taskListId, String taskId, String subtaskId, StringCallback callback) {
        OkGo.<String>get(RECORD_LIST_URL)
                .tag(context)
                .params("taskListId", taskListId)
                .params("taskId", taskId)
                .params("subtaskId", subtaskId)
                .execute(callback);
    }

    public static void addRecord(Context context, String taskListId, String taskId, String subtaskId, String recordId, long timestamp, StringCallback callback) {
        OkGo.<String>post(RECORD_URL)
                .tag(context)
                .params("taskListId", taskListId)
                .params("taskId", taskId)
                .params("subtaskId", subtaskId)
                .params("recordId", recordId)
                .params("timestamp", timestamp)
                .isMultipart(true)
                .execute(callback);
    }

    public static void deleteRecord(Context context, String taskListId, String taskId, String subtaskId, String recordId, StringCallback callback) {
        OkGo.<String>delete(RECORD_URL)
                .tag(context)
                .params("taskListId", taskListId)
                .params("taskId", taskId)
                .params("subtaskId", subtaskId)
                .params("recordId", recordId)
                .isMultipart(true)
                .execute(callback);
    }

    public static void uploadRecordFile(Context context, File file, int fileType, String taskListId, String taskId, String subtaskId, String recordId, long timestamp, StringCallback callback) {
        OkGo.<String>post(RECORD_FILE_URL)
                .tag(context)
                .params("file", file)
                .params("fileType", fileType)
                .params("taskListId", taskListId)
                .params("taskId", taskId)
                .params("subtaskId", subtaskId)
                .params("recordId", recordId)
                .params("timestamp", timestamp)
                .isMultipart(true)
                .execute(callback);
    }
}
