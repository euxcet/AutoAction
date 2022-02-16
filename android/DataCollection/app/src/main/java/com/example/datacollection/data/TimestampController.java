package com.example.datacollection.data;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class TimestampController {
    private List<Long> data = new ArrayList<>();
    private File saveFile;

    public TimestampController() {
    }

    public void start(File file) {
        this.saveFile = file;
    }

    public void stop() {
        data.clear();
    }

    public void add(long timestamp) {
        data.add(timestamp);
    }
}
