package com.example.contextactionlibrary.collect.trigger;

import android.content.Context;
import android.os.Build;
import android.util.Pair;

import androidx.annotation.RequiresApi;

import com.example.contextactionlibrary.collect.collector.Collector;
import com.example.contextactionlibrary.collect.data.Data;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;

public class OnceTrigger extends Trigger {

    public OnceTrigger(Context context, List<CollectorType> types) {
        super(context, types);
    }

    public OnceTrigger(Context context, CollectorType type) {
        super(context, type);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public List<Pair<String, CompletableFuture<Data>>> triggerAsync() {
        List<Pair<String, CompletableFuture<Data>>> futures = new LinkedList<>();
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        collectors.forEach(collector -> {
            // TODO: correct?
            collector.setSavePath(timestamp);
            Pair<String, CompletableFuture<Data>> pair = new Pair<>(collector.getClass().getName(), collector.collect());
            futures.add(pair);
        });
        return futures;
    }

    @Override
    public void trigger() {
    }

    @Override
    public String getName() {
        return "Data/Click";
    }

    public String getRecentIMUPath() {
        for (Collector collector: collectors) {
            if (collector.getSaveFolderName().equals("CompleteIMU")) {
                return collector.getRecentPath();
            }
        }
        return "";
    }
}
