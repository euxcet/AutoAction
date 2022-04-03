package com.hcifuture.shared.communicate;

public enum BuiltInContextEnum {
    Proximity,
    Table,
    Informational;

    public static BuiltInContextEnum fromString(String context) {
        switch (context) {
            case "Proximity":
                return Proximity;
            case "Table":
                return Table;
            case "Informational":
                return Informational;
            default:
                return null;
        }
    }
}
