package com.hcifuture.contextactionlibrary.collect.data;

import java.text.SimpleDateFormat;
import java.util.Date;

public class InformationData extends Data{
    String task;
    String type;
    Date time;

    public InformationData()
    {}

    public InformationData(String task,String type,Date time)
    {
        this.task = task;
        this.type = type;
        this.time = time;
    }
    public SimpleDateFormat formatter = new SimpleDateFormat("yyyyMMdd_HHmmss");


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
