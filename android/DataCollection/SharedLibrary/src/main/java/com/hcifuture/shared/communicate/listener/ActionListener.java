package com.hcifuture.shared.communicate.listener;

import com.hcifuture.shared.communicate.result.ActionResult;

/**
 * Get ActionResult and do something.
 */
public interface ActionListener {
    void onAction(ActionResult action);
}
