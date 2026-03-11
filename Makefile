CC = gcc
CFLAGS = -O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude -fopenmp
LDFLAGS = -lm -fopenmp
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

# Reseta o ambiente da pipeline: limpa build, modelos temporarios e resultados CSV
# Preserva: best_model.bin e logs historicos (*.txt)
clean:
	rm -rf $(BUILD_DIR)
	rm -f models/mlp_fold*.bin models/norm_fold*.bin models/selected_fold*.bin
	rm -f results/*.csv

.PHONY: all clean extract train test full
