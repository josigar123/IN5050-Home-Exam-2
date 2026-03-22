# Paths
BUILD   := build
GOLDEN  := golden
OUT     := out
REPORTS := reports
INPUT   ?= /mnt/sdcard/cipr

# Tools
PLAYER := mplayer
SHELL  := /bin/bash

# Binaries
ENC        := $(BUILD)/c63enc
ENC_GOLDEN := $(GOLDEN)/c63enc
DEC        := $(GOLDEN)/c63dec
PRED       := $(GOLDEN)/c63pred

# Common knobs
FRAMES ?= 10
SRC    ?= exp                  # exp | golden

# Inputs
FOREMAN_IN := $(INPUT)/foreman.yuv
TRACTOR_IN := $(INPUT)/tractor.yuv

# Playback dimensions
FOREMAN_RAW := w=352:h=288:format=i420
TRACTOR_RAW := w=1920:h=1080:format=i420

# Bitstreams
FOREMAN_C63_EXP    := $(OUT)/foreman.c63
TRACTOR_C63_EXP    := $(OUT)/tractor.c63
FOREMAN_C63_GOLDEN := $(OUT)/foreman_golden.c63
TRACTOR_C63_GOLDEN := $(OUT)/tractor_golden.c63

# Source selection for decode/pred
ifeq ($(SRC),golden)
  FOREMAN_C63_IN := $(FOREMAN_C63_GOLDEN)
  TRACTOR_C63_IN := $(TRACTOR_C63_GOLDEN)
  SRC_TAG := golden
else
  FOREMAN_C63_IN := $(FOREMAN_C63_EXP)
  TRACTOR_C63_IN := $(TRACTOR_C63_EXP)
  SRC_TAG := exp
endif

# Decode/pred outputs
FOREMAN_DEC_YUV  := $(OUT)/foreman_dec_$(SRC_TAG).yuv
TRACTOR_DEC_YUV  := $(OUT)/tractor_dec_$(SRC_TAG).yuv
FOREMAN_PRED_YUV := $(OUT)/foreman_pred_$(SRC_TAG).yuv
TRACTOR_PRED_YUV := $(OUT)/tractor_pred_$(SRC_TAG).yuv

# nsys profiling flags
SAMPLE ?= cpu
NSYS_FLAGS := --trace=nvtx,osrt \
              --sample=$(SAMPLE) \
              --backtrace=dwarf \
              --cpuctxsw=none \
              --stats=true \
              --force-overwrite true

# Default NSYS profile name
D_PROFILE_NAME := c63_profile

.PHONY: all help check build dirs \
        nsys-foreman nsys-tractor \
        encode-foreman encode-tractor encode-all \
        golden-encode-foreman golden-encode-tractor golden-encode-all \
        decode-foreman decode-tractor decode-all \
        pred-foreman pred-tractor pred-all \
        run-foreman run-tractor run-all \
        play-foreman-dec play-tractor-dec play-foreman-pred play-tractor-pred \
		calculate-psnr-foreman \
        clean clean-all

all: run-foreman

help:
	@echo "Build/Run:"
	@echo "  make build"
	@echo "  make encode-foreman FRAMES=10"
	@echo "  make golden-encode-foreman FRAMES=10"
	@echo "  make decode-foreman SRC=exp|golden"
	@echo "  make pred-foreman SRC=exp|golden"
	@echo "  make run-foreman SRC=exp|golden"

check:
	@command -v cmake >/dev/null || (echo "Missing: cmake"; exit 1)
	@command -v make >/dev/null || (echo "Missing: make"; exit 1)
	@command -v $(PLAYER) >/dev/null || (echo "Missing: $(PLAYER)"; exit 1)
	@test -x $(ENC_GOLDEN) || (echo "Missing or not executable: $(ENC_GOLDEN)"; exit 1)
	@test -x $(DEC) || (echo "Missing or not executable: $(DEC)"; exit 1)
	@test -x $(PRED) || (echo "Missing or not executable: $(PRED)"; exit 1)

build:
	cmake -B $(BUILD) && make -C $(BUILD)

dirs:
	mkdir -p $(OUT) $(REPORTS)

nsys-foreman: build dirs
	nsys profile $(NSYS_FLAGS) -o $(REPORTS)/$(D_PROFILE_NAME) \
	  $(ENC) -w 352 -h 288 -f $(FRAMES) -o $(FOREMAN_C63_EXP) $(FOREMAN_IN)

nsys-tractor: build dirs
	nsys profile $(NSYS_FLAGS) -o $(REPORTS)/$(D_PROFILE_NAME) \
	  $(ENC) -w 1920 -h 1080 -f $(FRAMES) -o $(TRACTOR_C63_EXP) $(TRACTOR_IN)

# Experimental encode
encode-foreman: build dirs
	$(ENC) -w 352 -h 288 -f $(FRAMES) -o $(FOREMAN_C63_EXP) $(FOREMAN_IN)

encode-tractor: build dirs
	$(ENC) -w 1920 -h 1080 -f $(FRAMES) -o $(TRACTOR_C63_EXP) $(TRACTOR_IN)

encode-all: encode-foreman encode-tractor

# Golden encode
golden-encode-foreman: dirs
	$(ENC_GOLDEN) -w 352 -h 288 -f $(FRAMES) -o $(FOREMAN_C63_GOLDEN) $(FOREMAN_IN)

golden-encode-tractor: dirs
	$(ENC_GOLDEN) -w 1920 -h 1080 -f $(FRAMES) -o $(TRACTOR_C63_GOLDEN) $(TRACTOR_IN)

golden-encode-all: golden-encode-foreman golden-encode-tractor

# Decode/pred using selected SRC bitstream
# c63dec may return non-zero on EOF despite producing valid output.
decode-foreman: dirs
	-$(DEC) $(FOREMAN_C63_IN) $(FOREMAN_DEC_YUV)

decode-tractor: dirs
	-$(DEC) $(TRACTOR_C63_IN) $(TRACTOR_DEC_YUV)

decode-all: decode-foreman decode-tractor

pred-foreman: dirs
	$(PRED) $(FOREMAN_C63_IN) $(FOREMAN_PRED_YUV)

pred-tractor: dirs
	$(PRED) $(TRACTOR_C63_IN) $(TRACTOR_PRED_YUV)

pred-all: pred-foreman pred-tractor

run-foreman: decode-foreman pred-foreman
run-tractor: decode-tractor pred-tractor
run-all: decode-all pred-all

# Quality check
PSNR_FRAMES ?= 300
FOREMAN_DEC_EXP    := $(OUT)/foreman_dec_exp.yuv
FOREMAN_DEC_GOLDEN := $(OUT)/foreman_dec_golden.yuv

calculate-psnr-foreman: dirs
	$(ENC) -w 352 -h 288 -f $(PSNR_FRAMES) -o $(FOREMAN_C63_EXP) $(FOREMAN_IN)
	-$(DEC) $(FOREMAN_C63_EXP) $(FOREMAN_DEC_EXP)
	$(ENC_GOLDEN) -w 352 -h 288 -f $(PSNR_FRAMES) -o $(FOREMAN_C63_GOLDEN) $(FOREMAN_IN)
	-$(DEC) $(FOREMAN_C63_GOLDEN) $(FOREMAN_DEC_GOLDEN)
	ffmpeg -s 352x288 -pix_fmt yuv420p -i $(FOREMAN_DEC_EXP) \
	       -s 352x288 -pix_fmt yuv420p -i $(FOREMAN_DEC_GOLDEN) \
	       -lavfi psnr -f null -


# Playback
play-foreman-dec:
	$(PLAYER) -demuxer rawvideo -rawvideo $(FOREMAN_RAW) $(FOREMAN_DEC_YUV)

play-tractor-dec:
	$(PLAYER) -demuxer rawvideo -rawvideo $(TRACTOR_RAW) $(TRACTOR_DEC_YUV)

play-foreman-pred:
	$(PLAYER) -demuxer rawvideo -rawvideo $(FOREMAN_RAW) $(FOREMAN_PRED_YUV)

play-tractor-pred:
	$(PLAYER) -demuxer rawvideo -rawvideo $(TRACTOR_RAW) $(TRACTOR_PRED_YUV)

clean:
	rm -rf $(OUT)

clean-all: clean
	rm -rf $(BUILD)