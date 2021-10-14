APP = main
APP_SRCS = main.c fcn.c
APP_CFLAGS += -Os
APP_LDFLAGS += -Os

include $(RULES_DIR)/pmsis_rules.mk
