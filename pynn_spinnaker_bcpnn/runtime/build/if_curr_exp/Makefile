PYNN_APP = neuron_if_curr_exp

# Find PyNN SpiNNaker directory
PYNN_SPINNAKER_DIR := $(shell pynn_spinnaker_path)
PYNN_SPINNAKER_RUNTIME_DIR = $(PYNN_SPINNAKER_DIR)/spinnaker/runtime

# Build object list
SOURCES = $(PYNN_SPINNAKER_RUNTIME_DIR)/common/config.cpp \
	$(PYNN_SPINNAKER_RUNTIME_DIR)/common/bit_field.cpp \
	$(PYNN_SPINNAKER_RUNTIME_DIR)/common/profiler.cpp \
	$(PYNN_SPINNAKER_RUNTIME_DIR)/neuron_processor/neuron_processor.cpp \
	$(PYNN_SPINNAKER_RUNTIME_DIR)/neuron_processor/neuron_models/if_curr.cpp

# Add both current  directory (for config.h) and
# runtime directories (for standard PyNN SpiNNaker includes)
CFLAGS += -I $(CURDIR) -I $(PYNN_SPINNAKER_RUNTIME_DIR)

# Override directory APLX gets loaded into so it's within module
APP_DIR = ../../../binaries

# Include base Makefile
include $(PYNN_SPINNAKER_RUNTIME_DIR)/Makefile.depend