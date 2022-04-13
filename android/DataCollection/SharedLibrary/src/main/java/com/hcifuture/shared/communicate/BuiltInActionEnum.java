package com.hcifuture.shared.communicate;

public enum BuiltInActionEnum {
    TapTap,
    TopTap,
    Knock,
    Pocket;

    public static BuiltInActionEnum fromString(String context) {
        switch (context) {
            case "TapTap":
                return TapTap;
            case "TopTap":
                return TopTap;
            case "Knock":
                return Knock;
            case "Pocket":
                return Pocket;
            default:
                return null;
        }
    }
}
