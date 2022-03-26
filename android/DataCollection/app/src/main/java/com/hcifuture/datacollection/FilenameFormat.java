package com.hcifuture.datacollection;

import java.text.SimpleDateFormat;
import java.util.Date;

public class FilenameFormat {
    private final SimpleDateFormat dateFormat = new SimpleDateFormat("yyMMddHHmmss");

    private String pathName;

    private String timestampFilename;
    private String sensorFilename;
    private String microphoneFilename;

    private String mDate;
    private String mUser;
    private String mTask;
    private String mSubtask;


    private static FilenameFormat filenameFormat;

    public static FilenameFormat getInstance() {
        if (filenameFormat == null)
            filenameFormat = new FilenameFormat();
        return filenameFormat;
    }

    public void setPathName(String pathName) {
        this.pathName = pathName;
    }

    public String getPathName() {
        return this.pathName;
    }


    public void setFilename(String user, String task, String subtask) {
        mDate = dateFormat.format(new Date());
        mUser = user;
        mTask = task;
        mSubtask = subtask;
        timestampFilename = "Timestamp_" + user + "_" + task + "_" + subtask + "_" + mDate;
        sensorFilename = "Sensor_" + user + "_" + task + "_" + subtask + "_" + mDate;
        microphoneFilename = "Microphone_" + user + "_" + task + "_" + subtask + "_" + mDate;
    }

    public String getVideoFilename(int c) {
        return "Video_" + mUser + "_" + mTask + "_" + mSubtask + "_" + mDate + "_" + c;
    }

    public String getTimestampFilename() {
        return timestampFilename;
    }

    public String getSensorFilename() {
        return sensorFilename;
    }

    public String getMicrophoneFilename() {
        return microphoneFilename;
    }

}
