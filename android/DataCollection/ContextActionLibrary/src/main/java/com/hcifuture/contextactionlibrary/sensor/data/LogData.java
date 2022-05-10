package com.hcifuture.contextactionlibrary.sensor.data;

import android.text.TextUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LogData extends Data {
    private final List<String> logs;
    private int historyLength;

    public LogData(int historyLength) {
        logs = Collections.synchronizedList(new ArrayList<>());
        this.historyLength = historyLength;
    }

    public void addLog(String log) {
        synchronized (logs) {
            logs.add(log);
            checkSize();
        }
    }

    public void addLogs(List<String> newLogs) {
        synchronized (logs) {
            logs.addAll(newLogs);
            checkSize();
        }
    }

    private void checkSize() {
        int overflow = logs.size() - historyLength;
        if (overflow > 0) {
            eraseLog(overflow);
        }
    }

    public void eraseLog(int length) {
        synchronized (logs) {
            logs.subList(0, length).clear();
        }
    }

    public String getString() {
        synchronized (logs) {
            return TextUtils.join("\n", logs);
        }
    }

    public int getSize() {
        return logs.size();
    }

    public void clear() {
        synchronized (logs) {
            logs.clear();
        }
    }

    @Override
    public DataType dataType() {
        return DataType.LogData;
    }

    public LogData deepClone() {
        synchronized (logs) {
            LogData data = new LogData(historyLength);
            for (String l: logs) {
                data.addLog(l);
            }
            return data;
        }
    }
}
