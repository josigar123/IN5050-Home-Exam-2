# Paths
BUILD  := build
GOLDEN := golden
OUT    := out
INPUT  ?= /mnt/sdcard/cipr

# Tools
PLAYER := mplayer

# Binaries
ENC        := $(BUILD)/c63enc
ENC_GOLDEN := $(GOLDEN)/c63enc
DEC        := $(GOLDEN)/c63dec
PRED       := $(GOLDEN)/c63pred

# Controls
FRAMES ?= 10
SRC    ?= exp          # exp | golden

# Input videos
FOREMAN_IN := $(INPUT)/foreman.yuv
TRACTOR_IN := $(INPUT)/tractor.yuv

# Dimensions
FOREMAN_RAW := w=352:h=288:format=i420
TRACTOR_RAW := w=1920:h=1080:format=i420

# Bitstreams
FOREMAN_C63_EXP    := $(OUT)/foreman.c63
TRACTOR_C63_EXP    := $(OUT)/tractor.c63
FOREMAN_C63_GOLDEN := $(OUT)/foreman_golden.c63
TRACTOR_C63_GOLDEN := $(OUT)/tractor_golden.c63

# Select decode/pred input based on SRC
ifeq ($(SRC),golden)
  FOREMAN_C63_IN := $(FOREMAN_C63_GOLDEN)
  TRACTOR_C63_IN := $(TRACTOR_C63_GOLDEN)
  TAG := golden
else
  FOREMAN_C63_IN := $(FOREMAN_C63_EXP)
  TRACTOR_C63_IN := $(TRACTOR_C63_EXP)
  TAG := exp
endif

# Outputs from decode/pred
FOREMAN_DEC_YUV := $(OUT)/foreman_dec_$(TAG).yuv
TRACTOR_DEC_YUV := $(OUT)/tractor_dec_$(TAG).yuv
FOREMAN_PRED_YUV := $(OUT)/foreman_pred_$(TAG).yuv
TRACTOR_PRED_YUV := $(OUT)/tractor_pred_$(TAG).yuv

.PHONY: all help check build dirs \
        encode-foreman encode-tractor encode-all \
        golden-encode-foreman golden-encode-tractor golden-encode-all \
        decode-foreman decode-tractor decode-all \
        pred-foreman pred-tractor pred-all \
        run-foreman run-tractor run-all \
        play-foreman-dec play-tractor-dec play-foreman-pred play-tractor-pred \
        clean clean-all

all: run-foreman

help:
	@echo "Common usage:"
	@echo "  make build"
	@echo "  make encode-foreman FRAMES=10"
	@echo "  make golden-encode-foreman FRAMES=10"
	@echo "  make decode-foreman SRC=exp"
	@echo "  make decode-foreman SRC=golden"
	@echo "  make pred-foreman SRC=exp"
	@echo "  make pred-foreman SRC=golden"
	@echo "  make run-foreman SRC=exp"
	@echo "  make run-foreman SRC=golden"
	@echo "  make play-foreman-dec SRC=golden"
	@echo "  make clean | make clean-all"

check:
	@command -v cmake >/dev/null || (echo "Missing: cmake"; exit 1)
	@command -v make >/dev/null || (echo "Missing: make"; exit 1)
	@command -v $(PLAYER) >/dev/null || (echo "Missing: $(PLAYER)"; exit 1)
	@test -x $(ENC_GOLDEN) || (echo "Missing or not executable: $(ENC_GOLDEN)"; exit 1)
	@test -x $(DEC)        || (echo "Missing or not executable: $(DEC)"; exit 1)
	@test -x $(PRED)       || (echo "Missing or not executable: $(PRED)"; exit 1)

build:
	cmake -B $(BUILD) && make -C $(BUILD)

dirs:
	mkdir -p $(OUT)

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

# Decode / pred (source selected by SRC)
decode-foreman: dirs
	$(DEC) $(FOREMAN_C63_IN) $(FOREMAN_DEC_YUV)

decode-tractor: dirs
	$(DEC) $(TRACTOR_C63_IN) $(TRACTOR_DEC_YUV)

decode-all: decode-foreman decode-tractor

pred-foreman: dirs
	$(PRED) $(FOREMAN_C63_IN) $(FOREMAN_PRED_YUV)

pred-tractor: dirs
	$(PRED) $(TRACTOR_C63_IN) $(TRACTOR_PRED_YUV)

pred-all: pred-foreman pred-tractor

# Pipelines
run-foreman: decode-foreman pred-foreman
run-tractor: decode-tractor pred-tractor
run-all: decode-all pred-all

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