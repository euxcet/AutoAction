package com.hcifuture.shared.communicate;

public enum SensorType {
    IMU,
    AUDIO,
    VIDEO,
    LOCATION,
    PROXIMITY,
    ACCESSIBILITY,
    BROADCAST;

    public static SensorType fromString(String context) {
        switch (context) {
            case "IMU":
                return IMU;
            case "AUDIO":
                return AUDIO;
            case "VIDEO":
                return VIDEO;
            case "LOCATION":
                return LOCATION;
            case "PROXIMITY":
                return PROXIMITY;
            case "ACCESSIBILITY":
                return ACCESSIBILITY;
            case "BROADCAST":
                return BROADCAST;
            default:
                return null;
        }
    }
}
