package com.hcifuture.shared.communicate.listener;

import com.hcifuture.shared.communicate.result.ContextResult;

/**
 * Get ContextResult and do something.
 */
public interface ContextListener {
    void onContext(ContextResult context);
}
