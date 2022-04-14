package com.hcifuture.shared.communicate;

public enum BuiltInActionEnum {
    TapTap,
    TopTap,
    Knock,
    Flip,
    Close,
    Pocket;

    public static BuiltInActionEnum fromString(String context) {
        switch (context) {
            case "TapTap":
                return TapTap;
            case "TopTap":
                return TopTap;
            case "Knock":
                return Knock;
            case "Close":
                return Close;
            case "Flip":
                return Flip;
            case "Pocket":
                return Pocket;
            default:
                return null;
        }
    }
}
