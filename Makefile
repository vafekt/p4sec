
BMV2_SWITCH_EXE = simple_switch_grpc
NO_P4 = true
P4C_ARGS = --p4runtime-files $(basename $@).p4info

include ../../utils/Makefile
