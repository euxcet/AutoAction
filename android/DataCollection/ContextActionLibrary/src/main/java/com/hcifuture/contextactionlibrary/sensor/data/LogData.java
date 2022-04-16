package com.hcifuture.contextactionlibrary.sensor.data;

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LogData extends Data implements Cloneable {
    private final List<String> logs;
    private int historyLength;
    private int checkpoint = 0;

    public LogData(int historyLength) {
        logs = Collections.synchronizedList(new ArrayList<>());
        this.historyLength = historyLength;
    }

    public void addLog(String log) {
        synchronized (logs) {
            logs.add(log);
            while (logs.size() > historyLength) {
                logs.remove(0);
            }
        }
    }

    public void eraseLog() {
        synchronized (logs) {
            if (checkpoint > 0) {
                logs.subList(0, checkpoint).clear();
                checkpoint = 0;
            }
        }
    }

    public String getString() {
        synchronized (logs) {
            checkpoint = logs.size();
            StringBuilder result = new StringBuilder();
            for (String s : logs) {
                result.append(s).append("\n");
            }
            return result.toString();
        }
    }

    public void clear() {
        logs.clear();
    }

    @Override
    public DataType dataType() {
        return DataType.LogData;
    }

    @NonNull
    @Override
    public LogData clone() {
        synchronized (logs) {
            LogData data = new LogData(historyLength);
            for (String l: logs) {
                data.addLog(l);
            }
            return data;
        }
    }
}
