
BMV2_SWITCH_EXE = simple_switch_grpc
P4C_ARGS = --p4runtime-files $(basename $@).p4info

include ../../utils/Makefile

# Skip compiling the Tofino-only P4 program (requires tna.p4, not available on BMv2)
p4sec_tofino.json: ;
