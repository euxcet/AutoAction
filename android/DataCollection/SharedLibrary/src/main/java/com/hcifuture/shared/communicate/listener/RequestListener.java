package com.hcifuture.shared.communicate.listener;

import com.hcifuture.shared.communicate.config.RequestConfig;
import com.hcifuture.shared.communicate.result.RequestResult;

/**
 * Get RequestResult and do something.
 */
public interface RequestListener {
    RequestResult onRequest(RequestConfig config);
}
