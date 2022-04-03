package com.hcifuture.shared.communicate;

import java.security.KeyException;

public enum BuiltInActionEnum {
    TapTap,
    Knock;

    public static BuiltInActionEnum fromString(String context) {
        switch (context) {
            case "TapTap":
                return TapTap;
            case "Knock":
                return Knock;
            default:
                return null;
        }
    }
}
