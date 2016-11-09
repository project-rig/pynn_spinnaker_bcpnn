PYNN_APP = synapse_bcpnnsynapse

# Find PyNN SpiNNaker directory
PYNN_SPINNAKER_DIR := $(shell pynn_spinnaker_path)
PYNN_SPINNAKER_RUNTIME_DIR = $(PYNN_SPINNAKER_DIR)/spinnaker/runtime

# Build object list
PYNN_SOURCES = $(PYNN_SPINNAKER_RUNTIME_DIR)/synapse_processor/synapse_processor.cpp

RIG_CPP_COMMON_SOURCES = rig_cpp_common/config.cpp \
	rig_cpp_common/bit_field.cpp \
	rig_cpp_common/profiler.cpp

# Add both current  directory (for config.h) and
# runtime directory (for standard PyNN SpiNNaker includes)
CFLAGS += -I $(CURDIR) -I $(PYNN_SPINNAKER_RUNTIME_DIR)

ifdef GENERATE_CSV
	CFLAGS += -DGENERATE_CSV
endif

# Override directory APLX gets loaded into so it's within module
APP_DIR = ../../../binaries

# Include base Makefile
include $(PYNN_SPINNAKER_RUNTIME_DIR)/Makefile.common