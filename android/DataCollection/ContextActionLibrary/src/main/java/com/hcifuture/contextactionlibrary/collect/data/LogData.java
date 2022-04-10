package com.hcifuture.contextactionlibrary.collect.data;

import android.util.Log;

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
            while (logs.size() > historyLength) {
                logs.remove(0);
            }
        }
    }

    public String getString() {
        synchronized (logs) {
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
}
