PULP_APP = seizdetcnn
PULP_APP_SRCS = $(shell find src -type f -regex ".*\.c\(pp\)?" -not -path "./build/*")

PULP_CFLAGS += -O3 -g3 -w -I src

PULP_LDFLAGS += -lm

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk

linker_cp:
	@cp pulpissimo.ld build/pulpissimo-zeroriscy/$(PULP_APP).ld

info:
	@echo SRC: $(PULP_APP_SRCS)
