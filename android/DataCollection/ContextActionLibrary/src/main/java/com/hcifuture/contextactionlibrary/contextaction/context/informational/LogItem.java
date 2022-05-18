package com.hcifuture.contextactionlibrary.contextaction.context.informational;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;

public class LogItem {
    String task;
    String type;
    Date time;

    public LogItem()
    {}

    public LogItem(String task,String type,Date time)
    {
        this.task = task;
        this.type = type;
        this.time = time;
    }
    public static SimpleDateFormat formatter = new SimpleDateFormat("yyyyMMdd_HHmmssSSS");


    public Date getTime() {
        return time;
    }

    public String getType() {
        return type;
    }

    public String getTask() {
        return task;
    }

    public void setType(String type) {
        this.type = type;
    }

    public void setTask(String task) {
        this.task = task;
    }

    public void setTime(Date time) {
        this.time = time;
    }
}
