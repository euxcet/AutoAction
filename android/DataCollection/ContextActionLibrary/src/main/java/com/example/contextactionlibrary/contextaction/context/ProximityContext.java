package com.example.contextactionlibrary.contextaction.context;

import android.content.Context;
import android.util.Log;

import com.example.contextactionlibrary.data.Preprocess;

public class ProximityContext extends ContextBase {
    private String TAG = "ProximityContext";

    private Preprocess preprocess;

    private long lastRecognized = 0L;

    public ProximityContext(Context context, ContextListener contextListener, int seqLength, String[] contexts) {
        super(context, contextListener, seqLength, contexts);
        preprocess = Preprocess.getInstance();
    }

    @Override
    public synchronized void start() {
        if (isStarted) {
            Log.d(TAG, "Context is already started.");
            return;
        }
        isStarted = true;
    }

    @Override
    public synchronized void stop() {
        if (!isStarted) {
            Log.d(TAG, "Context is already stopped");
            return;
        }
        isStarted = false;
    }

    @Override
    public void getContext() {
        if (!isStarted)
            return;
        long tmp = preprocess.checkLastNear((long)(1.5 * 1e9), lastRecognized);
        if (tmp != -1) {
            lastRecognized = tmp;
            if (contextListener != null)
                contextListener.onContext(this, contexts[1]);
        }
    }
}
