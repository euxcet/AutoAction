package com.example.ncnnlibrary.communicate.listener;

import com.example.ncnnlibrary.communicate.config.RequestConfig;
import com.example.ncnnlibrary.communicate.result.RequestResult;

public interface RequestListener {
    RequestResult onRequest(RequestConfig config);
}
