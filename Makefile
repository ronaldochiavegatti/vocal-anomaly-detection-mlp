CC = gcc
CFLAGS = -O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude
LDFLAGS = -lm
SRC_DIR = src
BUILD_DIR = build
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
TARGET = $(BUILD_DIR)/vocal_detect

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

extract: $(TARGET)
	./$(TARGET) extract

train: $(TARGET)
	./$(TARGET) train

test: $(TARGET)
	./$(TARGET) test

full: $(TARGET)
	./$(TARGET) full

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean extract train test full
