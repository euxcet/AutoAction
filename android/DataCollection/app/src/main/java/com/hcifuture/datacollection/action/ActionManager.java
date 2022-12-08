package com.hcifuture.datacollection.action;

import android.util.Log;
import android.util.Pair;

import com.hcifuture.datacollection.inference.Inferencer;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class ActionManager {
    private static volatile ActionManager instance;
    private List<ActionWithObject> actions;

    private ActionManager() {
        actions = new ArrayList<>();
    }

    public static ActionManager getInstance() {
        if (instance == null) {
            synchronized (Inferencer.class) {
                if (instance == null) {
                    instance = new ActionManager();
                }
            }
        }
        return instance;
    }

    public void register(ActionWithObject action) {
        if (actions.contains(action)) {
            return;
        }
        actions.add(action);
    }

    public void unregister(ActionWithObject action) {
        actions.removeIf(a -> a == action);
    }

    public List<ActionWithObject> getActions() {
        return actions;
    }

    public ActionResult classify(float[] frame, ActionEnum actionEnum) {
        float minDistance = 10000.0f;
        ActionWithObject result = null;
        for (int i = 0; i < actions.size(); i++) {
            ActionWithObject action = actions.get(i);
            if (action.getAction() == actionEnum) {
                float distance = action.distance(frame);
                if (distance < minDistance) {
                    minDistance = distance;
                    result = action;
                }
            }
        }
        return new ActionResult(result, minDistance, System.currentTimeMillis());
    }

    public List<ActionWithObject> filterWithActionEnum(ActionEnum actionEnum) {
        return actions.stream()
                .filter((v) -> v.getAction() == actionEnum)
                .collect(Collectors.toList());
    }

    public static String encodeActions(List<ActionWithObject> actions) {
        StringBuilder result = new StringBuilder();
        for (ActionWithObject action: actions) {
            result.append(action.getName())
                    .append(" ")
                    .append(action.getAction())
                    .append("\n");
        }
        return result.toString();
    }
}
