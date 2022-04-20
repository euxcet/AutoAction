package com.hcifuture.contextactionlibrary.contextaction;

/*
Example:
{
    "context": [
        {
            "builtInContext": "Informational",
            "sensorType": ["ACCESSIBILITY", "BROADCAST"],
            "integerParamKey": [],
            "integerParamValue": [],
            "longParamKey": [],
            "longParamValue": [],
            "floatParamKey": [],
            "floatParamValue": [],
            "booleanParamKey": [],
			"booleanParamValue": []
        }
    ],
    "action": [
        {
            "builtInAction": "TapTap",
            "sensorType": ["IMU"],
            "integerParamKey": ["SeqLength"],
		    "integerParamValue": [50],
		    "longParamKey": [],
		    "longParamValue": [],
		    "floatParamKey": [],
			"floatParamValue": [],
            "booleanParamKey": [],
			"booleanParamValue": []
        }
    ],
    "listenedSystemActions": [
        "android.intent.action.AIRPLANE_MODE",
        "android.intent.action.SCREEN_OFF",
        "android.intent.action.SCREEN_ON",
        "android.bluetooth.device.action.ACL_CONNECTED"
    ],
    "listenedSystemURIs": [
        "content://settings/system",
        "content://settings/global"
    ],
    "overrideSystemActions": false,
    "overrideSystemURIs": false
}
 */

import java.util.List;

public class ContextActionConfigBean {
    private List<ContextConfigBean> context;
    private List<ActionConfigBean> action;
    private List<String> listenedSystemActions;
    private List<String> listenedSystemURIs;
    private boolean overrideSystemActions = false;
    private boolean overrideSystemURIs = false;

    public void setContext(List<ContextConfigBean> context) {
        this.context = context;
    }

    public void setAction(List<ActionConfigBean> action) {
        this.action = action;
    }

    public List<ActionConfigBean> getAction() {
        return action;
    }

    public List<ContextConfigBean> getContext() {
        return context;
    }

    public List<String> getListenedSystemActions() {
        return listenedSystemActions;
    }

    public List<String> getListenedSystemURIs() {
        return listenedSystemURIs;
    }

    public boolean isOverrideSystemActions() {
        return overrideSystemActions;
    }

    public boolean isOverrideSystemURIs() {
        return overrideSystemURIs;
    }

    public static class ContextConfigBean {
        private String builtInContext;
        private List<String> sensorType;
        private List<String> integerParamKey;
        private List<Integer> integerParamValue;
        private List<String> longParamKey;
        private List<Long> longParamValue;
        private List<String> floatParamKey;
        private List<Float> floatParamValue;
        private List<String> booleanParamKey;
        private List<Boolean> booleanParamValue;

        public List<Boolean> getBooleanParamValue() {
            return booleanParamValue;
        }

        public List<Float> getFloatParamValue() {
            return floatParamValue;
        }

        public List<Integer> getIntegerParamValue() {
            return integerParamValue;
        }

        public List<Long> getLongParamValue() {
            return longParamValue;
        }

        public List<String> getBooleanParamKey() {
            return booleanParamKey;
        }

        public List<String> getFloatParamKey() {
            return floatParamKey;
        }

        public List<String> getIntegerParamKey() {
            return integerParamKey;
        }

        public List<String> getLongParamKey() {
            return longParamKey;
        }

        public List<String> getSensorType() {
            return sensorType;
        }

        public String getBuiltInContext() {
            return builtInContext;
        }

        public void setBooleanParamKey(List<String> booleanParamKey) {
            this.booleanParamKey = booleanParamKey;
        }

        public void setBooleanParamValue(List<Boolean> booleanParamValue) {
            this.booleanParamValue = booleanParamValue;
        }

        public void setBuiltInContext(String builtInContext) {
            this.builtInContext = builtInContext;
        }

        public void setFloatParamKey(List<String> floatParamKey) {
            this.floatParamKey = floatParamKey;
        }

        public void setFloatParamValue(List<Float> floatParamValue) {
            this.floatParamValue = floatParamValue;
        }

        public void setIntegerParamKey(List<String> integerParamKey) {
            this.integerParamKey = integerParamKey;
        }

        public void setIntegerParamValue(List<Integer> integerParamValue) {
            this.integerParamValue = integerParamValue;
        }

        public void setLongParamKey(List<String> longParamKey) {
            this.longParamKey = longParamKey;
        }

        public void setLongParamValue(List<Long> longParamValue) {
            this.longParamValue = longParamValue;
        }

        public void setSensorType(List<String> sensorType) {
            this.sensorType = sensorType;
        }
    }

    public static class ActionConfigBean {
        private String builtInAction;
        private List<String> sensorType;
        private List<String> integerParamKey;
        private List<Integer> integerParamValue;
        private List<String> longParamKey;
        private List<Long> longParamValue;
        private List<String> floatParamKey;
        private List<Float> floatParamValue;
        private List<String> booleanParamKey;
        private List<Boolean> booleanParamValue;

        public List<Boolean> getBooleanParamValue() {
            return booleanParamValue;
        }

        public List<Float> getFloatParamValue() {
            return floatParamValue;
        }

        public List<Integer> getIntegerParamValue() {
            return integerParamValue;
        }

        public List<Long> getLongParamValue() {
            return longParamValue;
        }

        public List<String> getBooleanParamKey() {
            return booleanParamKey;
        }

        public List<String> getFloatParamKey() {
            return floatParamKey;
        }

        public List<String> getIntegerParamKey() {
            return integerParamKey;
        }

        public List<String> getLongParamKey() {
            return longParamKey;
        }

        public List<String> getSensorType() {
            return sensorType;
        }

        public String getBuiltInAction() {
            return builtInAction;
        }

        public void setBooleanParamKey(List<String> booleanParamKey) {
            this.booleanParamKey = booleanParamKey;
        }

        public void setBooleanParamValue(List<Boolean> booleanParamValue) {
            this.booleanParamValue = booleanParamValue;
        }

        public void setBuiltInAction(String builtInAction) {
            this.builtInAction = builtInAction;
        }

        public void setFloatParamKey(List<String> floatParamKey) {
            this.floatParamKey = floatParamKey;
        }

        public void setFloatParamValue(List<Float> floatParamValue) {
            this.floatParamValue = floatParamValue;
        }

        public void setIntegerParamKey(List<String> integerParamKey) {
            this.integerParamKey = integerParamKey;
        }

        public void setIntegerParamValue(List<Integer> integerParamValue) {
            this.integerParamValue = integerParamValue;
        }

        public void setLongParamKey(List<String> longParamKey) {
            this.longParamKey = longParamKey;
        }

        public void setLongParamValue(List<Long> longParamValue) {
            this.longParamValue = longParamValue;
        }

        public void setSensorType(List<String> sensorType) {
            this.sensorType = sensorType;
        }
    }
}
