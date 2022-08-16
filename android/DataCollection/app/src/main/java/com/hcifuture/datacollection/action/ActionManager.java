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

    public Pair<Integer, Float> classify(float[] frame) {
        float min_distance = 10000.0f;
        int result = -1;
        for (int i = 0; i < actions.size(); i++) {
            ActionWithObject action = actions.get(i);
            float distance = action.distance(frame);
            if (distance < min_distance) {
                min_distance = distance;
                result = i;
            }
        }
        return new Pair<>(result, min_distance);
    }

    public List<ActionWithObject> filterWithActionEnum(ActionEnum actionEnum) {
        return actions.stream()
                .filter((v) -> v.getAction() == actionEnum)
                .collect(Collectors.toList());
    }

    public static String encodeActions(List<ActionWithObject> actions) {
        StringBuilder result = new StringBuilder();
        for (ActionWithObject action: actions) {
            result.append(action.getName()).append("\n");
        }
        return result.toString();
    }
}
