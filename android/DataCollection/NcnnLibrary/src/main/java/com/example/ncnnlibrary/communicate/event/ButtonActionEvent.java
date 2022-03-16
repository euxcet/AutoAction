package com.example.ncnnlibrary.communicate.event;

public class ButtonActionEvent {
    private String text;
    private String type;

    public ButtonActionEvent(String text, String type) {
        this.text = text;
        this.type = type;
    }

    public String getText() {
        return text;
    }

    public String getType() {
        return type;
    }
}
