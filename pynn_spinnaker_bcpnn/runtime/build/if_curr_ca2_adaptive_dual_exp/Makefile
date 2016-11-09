PYNN_APP = neuron_if_curr_ca2_adaptive_dual_exp

# Find PyNN SpiNNaker directory
PYNN_SPINNAKER_DIR := $(shell pynn_spinnaker_path)
PYNN_SPINNAKER_RUNTIME_DIR = $(PYNN_SPINNAKER_DIR)/spinnaker/runtime

PYNN_SPINNAKER_IF_CURR_CA2_ADAPTIVE_EXP_DIR := $(shell pynn_spinnaker_if_curr_ca2_adaptive_path)
PYNN_SPINNAKER_IF_CURR_CA2_ADAPTIVE_EXP_RUNTIME_DIR = $(PYNN_SPINNAKER_IF_CURR_CA2_ADAPTIVE_EXP_DIR)/runtime
PYNN_SPINNAKER_IF_CURR_DUAL_EXP_DIR := $(shell pynn_spinnaker_if_curr_dual_exp_path)
PYNN_SPINNAKER_IF_CURR_DUAL_EXP_RUNTIME_DIR = $(PYNN_SPINNAKER_IF_CURR_DUAL_EXP_DIR)/runtime

# Build object list
PYNN_SOURCES = $(PYNN_SPINNAKER_RUNTIME_DIR)/neuron_processor/neuron_processor.cpp

RIG_CPP_COMMON_SOURCES = rig_cpp_common/config.cpp \
	rig_cpp_common/bit_field.cpp \
	rig_cpp_common/profiler.cpp

# Add both current  directory (for config.h) and
# runtime directories (for standard PyNN SpiNNaker includes and ca2 adaptive includes)
CFLAGS += -I $(CURDIR) -I $(PYNN_SPINNAKER_RUNTIME_DIR) -I $(PYNN_SPINNAKER_IF_CURR_CA2_ADAPTIVE_EXP_RUNTIME_DIR) -I $(PYNN_SPINNAKER_IF_CURR_DUAL_EXP_RUNTIME_DIR)

# Override directory APLX gets loaded into so it's within module
APP_DIR = ../../../binaries

# Include base Makefile
include $(PYNN_SPINNAKER_RUNTIME_DIR)/Makefile.common