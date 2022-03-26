package com.hcifuture.datacollection.utils;

import java.util.Random;

public class RandomUtils {
    private static final String ID_ALLOWED_CHARACTERS = "0123456789qwertyuiopasdfghjklzxcvbnm";

    public static String generateRandomId(int size) {
        Random random = new Random();
        StringBuilder builder = new StringBuilder(size);
        for(int i = 0; i < size; i++) {
            builder.append(ID_ALLOWED_CHARACTERS.charAt(random.nextInt(ID_ALLOWED_CHARACTERS.length())));
        }
        return builder.toString();
    }

    public static String generateRandomTaskListId() {
        return "TL" + generateRandomId(8);
    }

    public static String generateRandomTaskId() {
        return "TK" + generateRandomId(8);
    }

    public static String generateRandomSubtaskId() {
        return "ST" + generateRandomId(8);
    }

    public static String generateRandomRecordId() {
        return "RD" + generateRandomId(8);
    }

    public static String generateRandomTrainId() {
        return "XT" + generateRandomId(8);
    }
}
