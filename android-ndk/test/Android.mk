LOCAL_PATH := $(call my-dir)

# hello
include $(CLEAR_VARS)
LOCAL_MODULE := hello
TARGET_ARCH := arm64
TARGET_PLATFORM := android-26
TARGET_ARCH_ABI := arm64-v8a
TARGET_ABI := android-26-arm64-v8a

LOCAL_C_INCLUDES := include
MY_CPP_LIST := $(wildcard $(LOCAL_PATH)/src/*.cpp)
MY_CC_LIST := $(wildcard $(LOCAL_PATH)/src/*.cc)

ifeq ($(TARGET_ARCH), arm64)
	LOCAL_SRC_FILES += $(MY_CPP_LIST:$(LOCAL_PATH)/%=%)
	LOCAL_SRC_FILES += $(MY_CC_LIST:$(LOCAL_PATH)/%=%)
endif

LOCAL_CFLAGS := -std=c++11 -DHAVE_PTHREADS
LOCAL_LDFLAGS := -llog
include $(BUILD_SHARED_LIBRARY)