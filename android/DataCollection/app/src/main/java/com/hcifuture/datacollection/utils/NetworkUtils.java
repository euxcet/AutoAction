package com.hcifuture.datacollection.utils;

import android.content.Context;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.google.gson.Gson;
import com.lzy.okgo.OkGo;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.callback.StringCallback;

import java.io.File;

/**
 * Defines some interface constants and data request methods to interact with the backend.
 */
public class NetworkUtils {
    private static final String TAG = "NetworkUtils";
    private static final String ROOT_URL = BuildConfig.WEB_SERVER;
    private static final String ALL_TASKLIST_URL = ROOT_URL + "/all_taskList";
    private static final String TASKLIST_HISTORY_URL = ROOT_URL + "/taskList_history";
    private static final String TASKLIST_URL = ROOT_URL + "/taskList";
    private static final String RECORD_LIST_URL = ROOT_URL + "/record_list";
    private static final String RECORD_URL = ROOT_URL + "/record";
    private static final String RECORD_FILE_URL = ROOT_URL + "/record_file";
    private static final String SAMPLE_NUMBER_URL = ROOT_URL + "/sample_number";
    private static final String SAMPLE_URL = ROOT_URL + "/sample";
    private static final String CUTTER_TYPE_URL = ROOT_URL + "/cutter_type";
    private static final String TRAIN_LIST_URL = ROOT_URL + "/train_list";
    private static final String TRAIN_URL = ROOT_URL + "/train";
    private static final String FILE_URL = ROOT_URL + "/file";
    private static final String MD5_URL = ROOT_URL + "/md5";
    private static final String TRAIN_FILE_URL = ROOT_URL + "/train_file";

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

    public static void updateTaskList(Context context, TaskListBean tasklist, long timestamp, StringCallback callback) {
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

    public static void addRecord(Context context, String taskListId, String taskId, String subtaskId,
                                 String userName, String recordId, long timestamp, StringCallback callback) {
        OkGo.<String>post(RECORD_URL)
                .tag(context)
                .params("taskListId", taskListId)
                .params("taskId", taskId)
                .params("subtaskId", subtaskId)
                .params("userName", userName)
                .params("recordId", recordId)
                .params("timestamp", timestamp)
                .isMultipart(true)
                .execute(callback);
    }

    public static void deleteRecord(Context context, String taskListId, String taskId,
            String subtaskId, String recordId, StringCallback callback) {
        OkGo.<String>delete(RECORD_URL)
                .tag(context)
                .params("taskListId", taskListId)
                .params("taskId", taskId)
                .params("subtaskId", subtaskId)
                .params("recordId", recordId)
                .isMultipart(true)
                .execute(callback);
    }

    public static void uploadRecordFile(Context context, File file, int fileType, String taskListId,
            String taskId, String subtaskId, String recordId, long timestamp, StringCallback callback) {
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

    public static void downloadRecordFile(Context context, String taskListId, String taskId,
            String subtaskId, String recordId, int fileType, FileCallback callback) {
        OkGo.<File>get(RECORD_FILE_URL)
                .tag(context)
                .params("taskListId", taskListId)
                .params("taskId", taskId)
                .params("subtaskId", subtaskId)
                .params("recordId", recordId)
                .params("fileType", fileType)
                .execute(callback);
    }

    public static void getCutterType(Context context, StringCallback callback) {
        OkGo.<String>get(CUTTER_TYPE_URL)
                .tag(context)
                .execute(callback);
    }

    public static void getTrainList(Context context, StringCallback callback) {
        OkGo.<String>get(TRAIN_LIST_URL)
                .tag(context)
                .execute(callback);
    }

    public static void startTrain(Context context, String trainId, String trainName, String taskListId, String taskIdList, long timestamp, StringCallback callback) {
        OkGo.<String>post(TRAIN_URL)
                .tag(context)
                .params("trainId", trainId)
                .params("trainName", trainName)
                .params("taskListId", taskListId)
                .params("taskIdList", taskIdList)
                .params("timestamp", timestamp)
                .isMultipart(true)
                .execute(callback);
    }

    public static void stopTrain(Context context, String trainId, long timestamp, StringCallback callback) {
        OkGo.<String>delete(TRAIN_URL)
                .tag(context)
                .params("trainId", trainId)
                .params("timestamp", timestamp)
                .isMultipart(true)
                .execute(callback);
    }

    public static void downloadFile(Context context, String filename, FileCallback callback) {
        OkGo.<File>get(FILE_URL)
                .tag(context)
                .params("filename", filename)
                .execute(callback);
    }

    public static void getMD5(Context context, String filename, StringCallback callback) {
        OkGo.<String>get(MD5_URL)
                .tag(context)
                .params("filename", filename)
                .execute(callback);
    }

    public static void downloadTrainLog(Context context, String trainId, FileCallback callback) {
        OkGo.<File>get(TRAIN_FILE_URL)
                .tag(context)
                .params("trainId", trainId)
                .params("fileType", "log")
                .execute(callback);
    }

    public static void downloadTrainMNNModel(String trainId, FileCallback callback) {
        OkGo.<File>get(TRAIN_FILE_URL)
                .params("trainId", trainId)
                .params("fileType", "mnn_model")
                .execute(callback);
    }

    public static void downloadTrainLabel(String trainId, FileCallback callback) {
        OkGo.<File>get(TRAIN_FILE_URL)
                .params("trainId", trainId)
                .params("fileType", "label")
                .execute(callback);
    }
}
