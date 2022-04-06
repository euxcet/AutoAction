package com.hcifuture.shared.communicate.listener;

import com.hcifuture.shared.communicate.result.ActionResult;

public interface ActionListener {
    void onActionRecognized(ActionResult action);
    void onAction(ActionResult action);
    void onActionSave(ActionResult action);
}
